"""查询计划与计划校验（P1 计划先行，2026-11-25）。

为什么要有这一层
----------------
已有的拦截全发生在**执行之后**：SQL 报错才重试、结果为空才放宽、结果非空但
条件被改掉才丢弃（`scope_guard`）。它们有个共同前提 —— 调用已经花掉了，
SQL 也已经写完了。

更难缠的错误发生在更早一步：**模型理解错了问题，然后按错误的理解写了一条
语法正确、结果合理的 SQL**。数字、列名、格式全对，用户不会怀疑。

所以这里先把"理解"变成一份**结构化计划**（指标 / 表 / 维度 / 过滤 / 时间 / 假设），
在生成 SQL 之前用代码校验三件代码能百分百确定的事：

1. **时间区间必须与数据覆盖有交集** —— 无交集就直接说没有数据，不许平移到有数据的月份；
2. **过滤取值必须逐字来自真实枚举** —— 对不上就澄清，不许挑"最接近的"；
3. **计划里的年份/月份必须与用户原话一致** —— 不许自己换年份。

表名/列名这类只做**提醒**不阻断：列名可能带表前缀、别名、大小写差异，
误拦的代价（把正常问题拦下来）比漏拦更大——与 `enum_guard` / `scope_guard` 同一原则。

本模块是纯函数，不依赖 LLM / 数据库 / 配置，可离线穷举测试。
"""
from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import date
from typing import Optional

from app.tools.scope_guard import extract_anchors, strip_code_fence

# 需要"先想清楚再写 SQL"的信号。都是为了压成本：简单问题（"各区域的销售额"）
# 直接生成就够，多一次调用只是白白烧 token。
# 「分别」「以及」不算过滤：它们是**多维度并列**（"各区域分别多少"），
# 属于最简单的一类聚合，出计划纯属浪费一次调用。实测把它们算进去会让
# 评测集一半的题都触发计划，token 白涨 50%。
_FILTER_SIGNALS = ("只", "仅", "除了", "不含", "剔除", "排除", "其中")
_RATIO_SIGNALS = ("占比", "比例", "百分比", "同比", "环比", "增长率", "平均",
                  "客单价", "单价", "人均")
# 超过这个长度的问句通常带多层限定，值得先出计划
LONG_QUESTION = 25

# 输出格式要求会把问句撑长，但它跟"问题难不难"毫无关系：
# 「各区域的销售额是多少？保留 2 位小数。」不该因为多了半句格式而多一次调用。
_FORMAT_NOISE = re.compile(
    r"(保留\s*\d+\s*位小数|用\s*[A-Za-z][A-Za-z0-9_-]*\s*格式(?:表示)?|"
    r"用百分比(?:数值)?表示|以百分比表示|用百分数表示)")

_YMD = re.compile(r"(?:19|20)\d{2}-\d{1,2}(?:-\d{1,2})?")


def needs_plan(question: str, has_history: bool = False, anchors: list = None,
               intent: Optional[dict] = None, route: str = "sql") -> tuple:
    """要不要为这个问题先出一份计划。返回 (bool, 原因)。

    判定故意**从宽到严**：只要有一条命中就出计划，因为计划的价值在于提前发现
    理解错误，而"这条问题到底难不难"本身就判断不准。真正压成本的是
    简单问题（短、单维度、无过滤）直接跳过。
    """
    if route != "sql":
        return False, "单表路线不启用（列少、代码不是声明式的，计划没法校验）"
    if intent:
        return True, "归因问题（要拆对比列与维度）"
    if has_history:
        return True, "多轮追问（省略说法容易改错对象）"
    if anchors:
        return True, "含相对时间（需要先把区间定下来）"

    text = (question or "").strip()
    for word in _FILTER_SIGNALS:
        if word in text:
            return True, f"含过滤语义（{word}）"
    for word in _RATIO_SIGNALS:
        if word in text:
            return True, f"含比率/派生指标（{word}）"
    core = _FORMAT_NOISE.sub("", text).strip()
    if len(core) > LONG_QUESTION:
        return True, f"问句较长（{len(core)} 字，可能有多层限定）"
    return False, "简单问题，直接生成"


# ---------------------------------------------------------------- 解析
def _as_date(value: str, end: bool = False) -> Optional[date]:
    """把计划里的日期写法统一成 date。

    只认 `YYYY-MM-DD` 与 `YYYY-MM`：后者按 end 决定取月初还是月末。
    认不出来就返回 None —— 宁可不校验，也不要拿一个猜出来的日期去拦人。
    """
    text = str(value or "").strip()
    match = _YMD.search(text)
    if not match:
        return None
    parts = match.group(0).split("-")
    try:
        year, month = int(parts[0]), int(parts[1])
        if len(parts) > 2:
            return date(year, month, int(parts[2]))
        return date(year, month, monthrange(year, month)[1] if end else 1)
    except ValueError:
        return None


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item not in (None, "")]
    return [value]


def _normalize_plan(data: dict) -> dict:
    """把模型给的 JSON 收拾成固定形状：缺字段补空、类型不对就丢。

    不因为某个字段写得怪就把整份计划丢掉 —— 剩下的字段仍能用来校验。
    """
    metric = data.get("metric")
    if not isinstance(metric, dict):
        metric = {}
    filters = []
    for item in _as_list(data.get("filters")):
        if isinstance(item, dict) and item.get("column"):
            filters.append({
                "column": str(item.get("column", "")),
                "values": [str(v) for v in _as_list(item.get("values"))],
                "op": str(item.get("op") or "in"),
            })
    time_range = data.get("time_range")
    if not isinstance(time_range, dict):
        time_range = {}
    return {
        "intent": str(data.get("intent") or "").strip(),
        "tables": [str(t) for t in _as_list(data.get("tables"))],
        "metric": {
            "name": str(metric.get("name") or "").strip(),
            "agg": str(metric.get("agg") or "").strip().upper(),
            "column": str(metric.get("column") or "").strip(),
        },
        "dimensions": [str(d) for d in _as_list(data.get("dimensions"))],
        "filters": filters,
        "time_range": {
            "column": str(time_range.get("column") or "").strip(),
            "start": str(time_range.get("start") or "").strip(),
            "end": str(time_range.get("end") or "").strip(),
        },
        "assumptions": [str(a) for a in _as_list(data.get("assumptions"))],
        "needs_clarify": str(data.get("needs_clarify") or "").strip(),
    }


def parse_plan(text: str) -> Optional[dict]:
    """从模型输出里取出计划 JSON。

    模型爱加代码围栏、爱在前面写一句"好的，计划如下："，所以取第一个 `{`
    到最后一个 `}` 之间的内容再解析，而不是要求整段就是 JSON。
    解析不出来返回 None —— 上层据此退回普通生成，不能让计划成为单点故障。
    """
    body = strip_code_fence(text or "").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return _normalize_plan(data)


# ---------------------------------------------------------------- 校验
def _issue(level: str, kind: str, message: str, **extra) -> dict:
    return {"level": level, "kind": kind, "message": message, **extra}


def _find_enum(enums: list, column: str) -> Optional[dict]:
    """按列名找枚举定义。计划里的列名通常不带表前缀。"""
    target = (column or "").strip().lower()
    if not target:
        return None
    for enum in enums or []:
        if str(enum.get("column", "")).lower() == target:
            return enum
    return None


def _months_between(start: date, end: date) -> set:
    """区间覆盖到的月份集合（用于判断用户说的"6 月"是否落在计划区间内）。"""
    months = set()
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.add(month if year == end.year else (year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def validate_plan(plan: dict, tables: dict, enums: list = None,
                  coverage: dict = None, question: str = "",
                  skip_anchor_check: bool = False) -> list:
    """校验计划与真实库、用户原话是否一致。返回问题清单（空表示没问题）。

    `tables` 是 `{表名: [列名]}`；`coverage` 是 `{min, max}`；
    `enums` 直接来自数据画像（真实取值，不是 catalog 里手写的）。

    `skip_anchor_check` 给归因问题用：归因比的是**两期**（"6 月比 5 月"），
    而计划的 `time_range` 只有一个区间，装不下两期。按单期校验会天天误拦，
    时段是否被换掉由 `attribution` 的时段解析与 `scope_guard` 负责。
    """
    issues: list = []
    tables = tables or {}
    all_columns = {str(c).lower() for cols in tables.values() for c in cols}

    # 1) 表名与列名：只提醒，不阻断（带前缀/别名的情况太多，误判率高）
    for name in plan.get("tables") or []:
        if name not in tables:
            issues.append(_issue(
                "warn", "table",
                f"计划里用到的表「{name}」不在库里（真实表：{'、'.join(tables) or '无'}），"
                f"生成时会忽略这个说法"))
    columns = [plan["metric"].get("column", "")] + list(plan.get("dimensions") or [])
    columns += [f.get("column", "") for f in plan.get("filters") or []]
    columns.append(plan.get("time_range", {}).get("column", ""))
    unknown_cols = sorted({c for c in columns if c and c.lower() not in all_columns})
    if unknown_cols:
        issues.append(_issue(
            "warn", "column",
            f"计划里的列「{'、'.join(unknown_cols)}」在 schema 里找不到，"
            f"生成时请改用真实列名"))

    # 2) 时间区间：与覆盖范围无交集是硬错误
    tr = plan.get("time_range") or {}
    start, end = _as_date(tr.get("start")), _as_date(tr.get("end"), end=True)
    cov_lo = _as_date((coverage or {}).get("min"))
    cov_hi = _as_date((coverage or {}).get("max"), end=True)
    if start and end:
        if end < start:
            issues.append(_issue("warn", "time",
                                 f"计划里的时间区间倒置（{start} ~ {end}），已忽略"))
        elif cov_lo and cov_hi and (end < cov_lo or start > cov_hi):
            issues.append(_issue(
                "fatal", "time",
                f"你问的时间段是 {start.isoformat()} ~ {end.isoformat()}，"
                f"库里只有 {cov_lo.isoformat()} ~ {cov_hi.isoformat()}，"
                f"两者没有交集。没有把时间平移到有数据的区间来凑数 —— "
                f"那样算出来的不是这个问题的答案。请把问题改成明确的年月再问。"))

    # 3) 过滤取值：必须逐字来自真实枚举，否则澄清
    for item in plan.get("filters") or []:
        enum = _find_enum(enums or [], item.get("column", ""))
        if not enum:
            continue
        allowed = [str(v) for v in (enum.get("values") or [])]
        bad = [v for v in item.get("values") or [] if v not in allowed]
        if bad:
            issues.append(_issue(
                "fatal", "enum",
                f"计划要按「{enum.get('label') or enum.get('column')}」筛选"
                f"「{'、'.join(bad)}」，但库里没有这个取值",
                payload={
                    "table": enum.get("table", ""),
                    "column": enum.get("column", ""),
                    "label": enum.get("label") or enum.get("column", ""),
                    "candidate": bad[0],
                    "values": allowed,
                }))

    # 4) 与用户原话对齐：年份/月份不许被换掉
    if question and not skip_anchor_check:
        anchors = extract_anchors(question, enums or [])
        user_years = {int(y) for y in anchors.get("years") or []}
        if start and end and user_years:
            span_years = {start.year, end.year}
            if not (user_years & span_years):
                issues.append(_issue(
                    "fatal", "anchor",
                    f"你问的是 {'、'.join(str(y) for y in sorted(user_years))} 年，"
                    f"计划里的时间却落在 {start.isoformat()} ~ {end.isoformat()}。"
                    f"没有拿别的年份的数据来代替 —— 那是换了个问题。"))
        user_months = set(anchors.get("months") or [])
        if start and end and user_months:
            span_months = _months_between(start, end)
            # 跨年时月份都用 (年, 月)，逐月比对更稳
            flat = {m if isinstance(m, int) else m[1] for m in span_months}
            missing = sorted(user_months - flat)
            if missing:
                issues.append(_issue(
                    "fatal", "anchor",
                    f"你问的是 {'、'.join(str(m) for m in missing)} 月，"
                    f"计划里的时间区间（{start.isoformat()} ~ {end.isoformat()}）"
                    f"不包含这个月份。"))

    # 5) 模型自己标注的假设：不阻断，但要让人看见
    for text in plan.get("assumptions") or []:
        issues.append(_issue("warn", "assumption", text))

    return issues


def has_fatal(issues: list) -> bool:
    return any(item.get("level") == "fatal" for item in issues)


def format_plan_hint(plan: dict, issues: list = None) -> str:
    """把计划拼成生成阶段的引导语。

    放在**当前问题之后**（而不是 system 里）：越靠近问题影响力越大，
    与 lesson 注入的位置选择同理。
    """
    if not plan:
        return ""
    lines = ["【查询计划（系统已校验，请严格按此写 SQL）】"]
    if plan.get("intent"):
        lines.append(f"- 意图：{plan['intent']}")
    metric = plan.get("metric") or {}
    if metric.get("name") or metric.get("column"):
        agg = metric.get("agg") or "SUM"
        column = metric.get("column") or "?"
        lines.append(f"- 指标：{agg}({column})"
                     + (f" → 「{metric['name']}」" if metric.get("name") else ""))
    if plan.get("dimensions"):
        lines.append(f"- 分组：{'、'.join(plan['dimensions'])}")
    for item in plan.get("filters") or []:
        values = "、".join(f"'{v}'" for v in item.get("values") or [])
        op = (item.get("op") or "in").upper()
        lines.append(f"- 过滤：{item.get('column')} {op} ({values})")
    tr = plan.get("time_range") or {}
    if tr.get("start") and tr.get("end"):
        lines.append(f"- 时间：{tr.get('column') or '日期列'} 在 "
                     f"{tr['start']} ~ {tr['end']}（含端点）")
    if plan.get("tables"):
        lines.append(f"- 用到的表：{'、'.join(plan['tables'])}")
    for text in plan.get("assumptions") or []:
        lines.append(f"- 假设（要在回答里写明）：{text}")
    for item in issues or []:
        if item.get("level") == "warn" and item.get("kind") in ("table", "column"):
            lines.append(f"- 注意：{item['message']}")
    lines.append("上表中的表名、列名、取值、时间范围必须与用户问题一致，不要自行更改。")
    return "\n".join(lines)


def format_plan_summary(plan: dict) -> str:
    """一行摘要，给命令行与界面用（完整计划放在折叠区）。"""
    if not plan:
        return ""
    metric = plan.get("metric") or {}
    parts = []
    if metric.get("column"):
        parts.append(f"{metric.get('agg') or 'SUM'}({metric['column']})")
    if plan.get("dimensions"):
        parts.append(f"按 {'、'.join(plan['dimensions'])} 分组")
    tr = plan.get("time_range") or {}
    if tr.get("start") and tr.get("end"):
        parts.append(f"{tr['start']} ~ {tr['end']}")
    for item in plan.get("filters") or []:
        values = "、".join(str(v) for v in item.get("values") or [])
        parts.append(f"{item.get('column')} = {values}")
    if plan.get("tables"):
        parts.append(f"{len(plan['tables'])} 张表")
    return " · ".join(parts) or plan.get("intent") or ""
