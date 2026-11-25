"""两段式探查（P1，2026-11-25）：允许模型"先看一眼真实数据，再写最终 SQL"。

为什么需要它
------------
`catalog.yaml` 里声明的枚举列（区域、品类、客户等级）有取值校验兜着，
但库里还有很多**没写进词典的列**——状态、渠道、支付方式、客户名。
模型对这些列的取值只能猜：猜"已完成"，实际是"completed"；猜"华东"，实际有"华东区"。
猜错了不会报错，SQL 照样跑通，只是悄悄返回 0 行或 NULL —— 和 P0 那批"空结果自愈"
踩的是同一类坑，只是发生在更早的一步。

两条既有出路都不理想：
- `enum_guard` 只能拦**词典里声明过**的列，没声明的看不见；
- 打断用户去澄清（"你说的状态是哪个？"）体验差，而且用户往往也不知道库里怎么写的。

所以给模型一个只读的"看一眼"通道：它先说要看什么，系统执行后把真实取值回灌，
再让它写最终 SQL。这正是 Vanna `intermediate_sql` 的用法。

限制（宁可严一点）
------------------
探查是"看一眼取值"，不是"跑个查询"：
- 只允许一条 `SELECT [DISTINCT] 列, ... FROM 一张表 [WHERE/GROUP BY/ORDER BY/LIMIT]`；
- 禁止 JOIN / UNION / 子查询 / WITH / 聚合函数 / `SELECT *`，列必须是纯列名，最多 4 列；
- 执行时最多取 50 行、每列最多回灌 30 个取值、每个值截断到 30 字符。

严的价值在于**可解释**：一个被拒的探查，用户能一眼看懂为什么被拒；
而"允许任意只读查询"等于把整库的扫描能力开放给一次顺手的调用。

失败时的姿态
------------
- 探查语句被拒 / 执行报错：**把原因回灌**给模型（"没有这一列"恰恰是最有用的信息），
  让它据此改写；额度照样算用掉；
- 重新生成失败：返回空语句，交给重试环，而不是把 `PEEK:` 那行当 SQL 去执行；
- 额度用完（每次问答 1 次）：明确告诉模型"不能再探了"，再让它写一次，避免死循环。

与计划先行（F2.10）的分工
-------------------------
计划先行管**意图**（要算哪个指标、按什么维度、时间对不对），在生成前就用代码校验；
探查管**事实**（这一列到底有哪些值）。前者防"理解错"，后者防"猜错字面量"。
"""
import re
import time
from typing import Any, Optional

import pandas as pd

from app.tools.sql_tool import UnsafeSQLError, check_sql_safety, run_sql

MAX_PEEKS = 1                  # 每次问答最多探查一次，防死循环
PEEK_MAX_ROWS = 50             # 执行时最多取多少行
PEEK_MAX_COLUMNS = 4           # 一次最多看几列
PEEK_MAX_VALUES = 30           # 单列最多回灌几个取值
PEEK_MAX_ROWS_SHOWN = 20       # 多列时最多回灌几行
PEEK_MAX_VALUE_CHARS = 30      # 单个取值的字符上限

PEEK_PREFIX = "PEEK:"

# 模型可能写成 `PEEK:` 也可能写成全角冒号，两种都认
_PEEK_RE = re.compile(r"^[ \t]*peek[ \t]*[:：][ \t]*(.+?)[ \t]*$", re.I | re.M)
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_COLUMN_RE = re.compile(rf"^(?:{_IDENT}\s*\.\s*)?{_IDENT}$")
_FROM_TABLE_RE = re.compile(rf"\bfrom\s+({_IDENT})", re.I)
_CLAUSE_WORDS = {"where", "group", "order", "limit", "having", "window", "on", "join"}

# 回灌给模型的引导语。为什么要把"别拿相近的值顶替"写在这里：
# 探查的目的就是消除猜测，如果模型看到"办公用品"不存在还去改用"配件"，
# 这次多花的调用反而制造了一个更难发现的错误（同 P0-2b）。
PEEK_HINT_HEADER = "【探查结果】以下是执行你的探查语句后拿到的真实数据："
PEEK_HINT_FOOTER = (
    "请据此写出最终 SQL。如果探查结果显示你设想的值并不存在，不要拿相近的值顶替："
    "按真实取值改写，或者输出 DEFER: <缺少什么字段或表>。"
)
PEEK_FAIL_TEMPLATE = (
    "【探查未成功】{reason}\n"
    "探查额度已用完，不要再输出 PEEK。请直接按已知的库结构写最终 SQL；"
    "实在写不出来就输出 DEFER: <缺少什么字段或表>。"
)
PEEK_EXHAUSTED_HINT = (
    "【探查额度已用完】本次问答只能探查一次，系统没有执行你刚才的探查。\n"
    "请直接按已知的库结构写最终 SQL；实在写不出来就输出 DEFER: <缺少什么字段或表>。"
)
PEEK_OFF_NOTE = "（本次不需要探查：不要输出 PEEK，直接给出最终 SQL。）"


class PeekError(Exception):
    """探查语句不符合白名单。"""


# ---------------------------------------------------------------- 解析

def parse_peek(text: str) -> Optional[str]:
    """从模型输出里抠出探查语句；没有则返回 None。

    只取第一条：真要给多条，说明模型在拿探查当查询用，不该纵容。
    """
    match = _PEEK_RE.search(text or "")
    if not match:
        return None
    sql = match.group(1).strip().rstrip(";").strip()
    return sql or None


def strip_peek(text: str) -> str:
    """去掉 `PEEK:` 行，留下其余内容（额度用完或二次探查时的兜底）。"""
    lines = [ln for ln in (text or "").splitlines() if not _PEEK_RE.match(ln)]
    return "\n".join(lines).strip()


def _table_of(sql: str) -> str:
    match = _FROM_TABLE_RE.search(sql or "")
    return match.group(1) if match else ""


# ---------------------------------------------------------------- 安全白名单

def check_peek_sql(sql: str) -> str:
    """校验探查语句，返回可执行版本；不合格抛 PeekError。

    先用 `check_sql_safety` 过黑名单与"单条 SELECT"（与正式执行同一套底线），
    再用白名单把形态收窄到"看一眼取值"。两层都要有：
    黑名单挡已知危险，白名单挡"没预料到但越权"的写法。
    """
    try:
        body = check_sql_safety(sql)
    except UnsafeSQLError as e:
        raise PeekError(str(e)) from e

    if re.match(r"\s*with\b", body, re.I):
        raise PeekError("探查语句不要用 WITH，直接写 SELECT ... FROM ...")

    match = re.match(r"\s*select\s+(?P<cols>.+?)\s+from\s+(?P<rest>.+)$",
                     body, re.I | re.S)
    if not match:
        raise PeekError("探查语句必须是 SELECT 列 FROM 表 的形式")

    cols, rest = match.group("cols"), match.group("rest")
    cols = re.sub(r"^\s*distinct\s+", "", cols, flags=re.I).strip()

    if "*" in cols:
        raise PeekError("探查请点名列名，不要用 *")
    if "(" in cols or ")" in cols:
        raise PeekError("探查只列列名，不要用函数或聚合（如 COUNT / strftime）")

    names = [item.strip() for item in cols.split(",") if item.strip()]
    if not names:
        raise PeekError("探查语句没有指定列")
    if len(names) > PEEK_MAX_COLUMNS:
        raise PeekError(f"探查最多看 {PEEK_MAX_COLUMNS} 列（这次有 {len(names)} 列）")
    for name in names:
        if not _COLUMN_RE.match(name):
            raise PeekError(f"只能探查纯列名，`{name[:40]}` 不是合法列名")

    table_match = re.match(rf"^\s*({_IDENT})", rest)
    if not table_match:
        raise PeekError(f"探查语句只能查一张表（表名要在 FROM 之后）")
    after = rest[table_match.end():]
    alias_match = re.match(rf"^\s+(?:as\s+)?({_IDENT})\b", after, re.I)
    if alias_match and alias_match.group(1).lower() not in _CLAUSE_WORDS:
        after = after[alias_match.end():]
    if not re.match(r"^\s*(where|group|order|limit|having|$)", after, re.I):
        raise PeekError("探查语句只允许查一张表，不能用 JOIN / UNION / 子查询或多表逗号")

    if re.search(r"\b(join|union|intersect|except)\b", rest, re.I):
        raise PeekError("探查语句不能用 JOIN / UNION")
    if re.search(r"\(\s*select\b", rest, re.I):
        raise PeekError("探查语句不能用子查询")

    return body


# ---------------------------------------------------------------- 执行与回灌

def run_peek(sql: str, db_path: str = None) -> dict:
    """执行探查语句，返回 {ok, df, table, columns, rows, truncated, error}。

    行数用 `run_sql` 的 fetchmany 限制（只读连接 + 查询超时一并继承），
    不靠给 SQL 补 LIMIT —— 模型自己写了 LIMIT 也不会被改动。
    """
    empty = {"ok": False, "df": None, "table": "", "columns": [],
             "rows": 0, "truncated": False, "error": ""}
    try:
        safe = check_peek_sql(sql)
    except PeekError as e:
        return {**empty, "error": str(e), "table": _table_of(sql)}

    result = run_sql(safe, db_path, max_rows=PEEK_MAX_ROWS)
    table = _table_of(safe)
    if not result["ok"]:
        return {**empty, "table": table,
                "error": f"{result.get('stage')} 阶段：{result.get('error')}"}

    df = result["df"]
    return {"ok": True, "df": df, "table": table,
            "columns": [str(c) for c in df.columns],
            "rows": len(df), "truncated": result.get("truncated", False),
            "error": ""}


def _fmt_value(value: Any) -> str:
    """把一个单元格转成给人看的短字符串：空值标注、整数去小数点、超长截断。"""
    if value is None:
        return "(空)"
    try:
        if pd.isna(value):
            return "(空)"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    if not text:
        return "(空)"
    if len(text) > PEEK_MAX_VALUE_CHARS:
        text = text[:PEEK_MAX_VALUE_CHARS] + "…"
    return text


def _unique_values(series) -> tuple:
    """一列的取值：去重保序，分开统计"空值"（SQLite 的 NULL 也是信息）。"""
    values, blanks = [], 0
    for item in series.tolist():
        text = _fmt_value(item)
        if text == "(空)":
            blanks += 1
            continue
        if text not in values:
            values.append(text)
    return values, blanks


def format_peek_result(result: dict) -> str:
    """把探查结果压成回灌给模型的文本。

    为什么不是把结果表整个塞回去：50 行 × 4 列的表格能吃掉上千 token，
    而模型真正需要的是"这一列有哪些取值"。单列给取值清单，多列才给行
    （此时行与行的组合关系本身是信息）。
    """
    if not result.get("ok"):
        reason = (result.get("error") or "未知错误").strip()
        return PEEK_FAIL_TEMPLATE.format(reason=reason)

    df = result.get("df")
    columns = result.get("columns") or []
    table = result.get("table") or ""
    truncated = bool(result.get("truncated"))

    if df is None or len(df) == 0:
        return (PEEK_HINT_HEADER + "\n"
                + f"{table or '该查询'} 在这个条件下没有任何数据。\n"
                + PEEK_HINT_FOOTER)

    if len(columns) == 1:
        values, blanks = _unique_values(df[columns[0]])
        line = f"{columns[0]} 共 {len(values)} 个不同取值："
        line += " / ".join(values[:PEEK_MAX_VALUES]) if values else "(全部为空)"
        if len(values) > PEEK_MAX_VALUES:
            line += f" …（还有 {len(values) - PEEK_MAX_VALUES} 个未显示）"
        if blanks:
            line += f"（另有 {blanks} 行为空）"
        if truncated:
            line += f"；只看了前 {PEEK_MAX_ROWS} 行"
        body = line
    else:
        head = f"{table}({', '.join(columns)}) 共 {len(df)} 行"
        head += "（行数已截断）" if truncated else ""
        lines = [head + "："]
        for _, row in df.head(PEEK_MAX_ROWS_SHOWN).iterrows():
            lines.append("  " + " | ".join(_fmt_value(row[c]) for c in columns))
        if len(df) > PEEK_MAX_ROWS_SHOWN:
            lines.append(f"  …（还有 {len(df) - PEEK_MAX_ROWS_SHOWN} 行未显示）")
        body = "\n".join(lines)

    return PEEK_HINT_HEADER + "\n" + body + "\n\n" + PEEK_HINT_FOOTER


def summarize_peek(result: dict) -> str:
    """一行给人看的说明（命令行 `[探查]`、网页上的小字）。"""
    if not result.get("ok"):
        return f"未成功：{(result.get('error') or '未知错误')[:80]}"

    df = result.get("df")
    columns = result.get("columns") or []
    table = result.get("table") or ""
    if df is None or len(df) == 0:
        return f"{table or '该查询'} 在这个条件下没有任何数据"

    if len(columns) == 1:
        values, _ = _unique_values(df[columns[0]])
        preview = " / ".join(values[:6])
        if len(values) > 6:
            preview += " …"
        name = f"{table}.{columns[0]}" if table else columns[0]
        return f"{name} → {len(values)} 个取值：{preview}"

    preview = " / ".join(" | ".join(_fmt_value(row[c]) for c in columns)
                         for _, row in df.head(3).iterrows())
    if len(df) > 3:
        preview += " …"
    head = f"{table}({', '.join(columns)})" if table else ", ".join(columns)
    return f"{head} → {len(df)} 行：{preview}"


# ---------------------------------------------------------------- 串起两段

def maybe_peek(ctx, text: str, regenerate, trace_id: str = "",
               remaining: int = MAX_PEEKS) -> tuple:
    """模型若要求探查，就执行一次、回灌结果并让它重新生成。

    返回 `(最终文本, 更新字段)`，更新字段可直接并进 state（也可放进 AskResult）：
    `peek_used / peek_rows / peek_note / peek_error / peek_ms`。

    regenerate 是回调（`hint -> 新文本`），两个引擎各自传自己的生成函数：
    这样"探查一轮"的流程只有一份实现，不会出现"状态机会探查、线性版不会"。
    """
    update = {"peek_used": False, "peek_rows": 0, "peek_note": "",
              "peek_error": "", "peek_ms": 0}
    sql = parse_peek(text or "")
    if not sql:
        return text, update

    update["peek_used"] = True
    started = time.perf_counter()

    if not getattr(ctx, "use_peek", True):
        # 关掉了却还收到 PEEK：去掉这行，让后续按普通情况处理（提示词里也会拦一道）
        update["peek_error"] = "探查已关闭"
        return strip_peek(text), update

    db_path = getattr(ctx, "db_path", None)
    if not db_path:
        # 没有库就没有可探查的东西（pandas 路线本就不该发 PEEK）
        update["peek_error"] = "当前数据源不支持探查"
        return strip_peek(text), update

    if remaining <= 0:
        note = PEEK_EXHAUSTED_HINT
        update["peek_error"] = "探查额度已用完"
    else:
        result = run_peek(sql, db_path)
        update["peek_rows"] = int(result.get("rows") or 0)
        update["peek_error"] = (result.get("error") or "")[:200]
        if result.get("ok"):
            update["peek_note"] = summarize_peek(result)
        note = format_peek_result(result)

    try:
        final = regenerate(note)
    except Exception as e:
        update["peek_ms"] = round((time.perf_counter() - started) * 1000)
        update["peek_error"] = update["peek_error"] or f"探查后重新生成失败：{type(e).__name__}"
        return "", update

    update["peek_ms"] = round((time.perf_counter() - started) * 1000)
    final = final or ""
    if parse_peek(final):
        # 第二次还在要探查 → 不执行，去掉这行，避免"探了又探"的死循环
        final = strip_peek(final)
        update["peek_error"] = update["peek_error"] or "模型在探查后仍要求再探查"
    return final, update
