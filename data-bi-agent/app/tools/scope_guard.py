"""范围守卫：不许可把用户明确给出的条件换掉（P0-2b）。

起因（真实评测依据，不是假想）：
问「2024 年 3 月的订单明细有哪些？」第一轮返回 0 行，触发一次"放宽条件"重试，
重试时模型把年份改成 `2025-03`，拿回 100 行真实数据，回答里一个字都没提 ——
用户以为看到的是 2024 年的明细，实际是 2025 年的。连续 3 次都这样改。

也就是说，「空结果自愈」这个本该让回答更诚实的机制，反过来成了编数据的入口。
更糟的是原来的提示词写着"放宽范围"，等于亲口邀请模型去改时间范围。

空结果的合理处置只有两种：
1. 修写法（日期格式、枚举取值写法）；
2. 如实回答"这个范围确实没有数据"。
改年份/改月份/删掉用户明确要求的取值，都不是放宽，是换了个问题。

提示词已同步收紧，但**提示词只能降概率**，所以这里再用代码兜一层：
重试生成的语句里，用户原话中的年份、季度、月份（含「3-5 月」这种区间）、
枚举取值必须仍然在。
"""
from __future__ import annotations

import re
from typing import List, Optional

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_MONTH_RE = re.compile(r"(?<!\d)(1[0-2]|0?[1-9])\s*月")
# 自定义跨度（「3-5 月」「3 月到 5 月」）。要求末尾有「月」，否则 '2025-06'
# 这类日期写法会被当成区间。只认数字月份：中文月份的跨度在归因里另有一套解析，
# 这里守的是"用户写下的范围有没有被换掉"，抓不到宁可放过（漏判优于误拦）。
_SPAN_RE = re.compile(
    r"(?<!\d)(1[0-2]|0?[1-9])\s*月?\s*[到至\-–—~～]\s*(1[0-2]|0?[1-9])\s*月")
# 季度：「Q2」「2025Q2」「2025-Q2」「第二季度」「第 2 季度」。
# 与 attribution.QUARTER_RE 是同一套写法（那边切时段，这边守范围），改一处要同步另一处，
# tests/test_scope.py 里有对拍用例盯着。
_QUARTER_RE = re.compile(
    r"(?:(?P<y>\d{4})\s*(?:年|[-/])?\s*)?"
    r"(?:第\s*(?P<q1>[1-4一二三四])\s*季度|Q\s*(?P<q2>[1-4])(?![\d]))",
    re.IGNORECASE,
)
_CN_QUARTER = {"一": 1, "二": 2, "三": 3, "四": 4}
# 裸月份数字的上下文（`BETWEEN 4 AND 6`、`month = 6`、`IN (4,5,6)`）。
# 必须挂在比较/区间关键词后面，孤零零的数字不算 —— `LIMIT 6` 不是六月。
_BARE_MONTH_CTX = r"(?:between|and|>=|<=|>|<|=|,|\bin\s*\()\s*0?%d(?!\d)"
_LIMIT_RE = re.compile(r"\blimit\s+[\d\s,]+", re.IGNORECASE)
_SQL_HEAD_RE = re.compile(r"(?is)^\s*(?:select|with)\b")
_DEFER_RE = re.compile(r"(?is)^\s*DEFER\b\s*[:：]?\s*(.+)$")


def strip_code_fence(text: str) -> str:
    """去掉模型爱加的 markdown 代码块围栏（```sql ... ```）。"""
    body = (text or "").strip()
    if body.startswith("```"):
        body = body[3:]
        # 围栏后通常紧跟语言标记和换行，去掉第一行
        body = body.split("\n", 1)[1] if "\n" in body else ""
    body = body.rstrip()
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def strip_sql_comments(text: str) -> str:
    """去掉整行的 `--` 注释，避免"以注释开头"被误判成不是 SQL。"""
    lines = [ln for ln in (text or "").splitlines() if not ln.strip().startswith("--")]
    return "\n".join(lines).strip()


def looks_like_sql(statement: str) -> bool:
    """语句是不是一条能执行的查询。

    模型有时会整段输出自然语言（"数据库中不存在退货相关字段……"），
    以前这种句子会被当成 SQL 丢给执行器报语法错，再重试就跑到别的问题上去了。
    """
    return bool(_SQL_HEAD_RE.match(strip_sql_comments(strip_code_fence(statement))))


def parse_defer(statement: str) -> Optional[str]:
    """解析 `DEFER: 原因` 约定 —— 模型主动说明"这题用现有数据答不了"。"""
    match = _DEFER_RE.match(strip_code_fence(statement))
    if not match:
        return None
    reason = match.group(1).strip()
    return reason or "模型判断这个问题无法用现有数据回答"


def extract_quarters(text: str) -> List[tuple]:
    """抽出原话里的季度，返回 [(年 或 None, 季度序号)]，按出现顺序、自动去重。

    季度不能只按字面守：用户写「Q2」，模型可以写成 `2025Q2`，也可以展开成
    `'2025-04'/'2025-05'/'2025-06'` 或日期区间 `>= '2025-04-01' AND < '2025-07-01'`，
    三种都算答对了。所以这里只负责"认出用户问的是哪一季度"，判是否还在交给
    `_quarter_present` 多写法比对。
    """
    found: List[tuple] = []
    for match in _QUARTER_RE.finditer(text or ""):
        quarter = _quarter_of(match)
        if not quarter:
            continue
        year = int(match.group("y")) if match.group("y") else None
        if (year, quarter) not in found:
            found.append((year, quarter))
    return found


def _quarter_of(match) -> Optional[int]:
    raw = match.group("q1") or match.group("q2")
    if raw is None:
        return None
    return int(raw) if str(raw).isdigit() else _CN_QUARTER.get(str(raw))


def extract_anchors(question: str, enums: Optional[list] = None) -> dict:
    """抽出用户原话里"不能动"的条件：年份、季度、月份区间、枚举取值。"""
    text = question or ""
    spans = [(int(a), int(b)) for a, b in _SPAN_RE.findall(text) if int(a) <= int(b)]
    # 跨度里的月份不再单算锚点：'3-5 月' 会命中「5 月」，若既按单月又按区间各查一遍，
    # 区间被正常展开成三个月反而会被报"找不到 5 月"（模型爱把上界写成下一个月 1 日）。
    rest = _SPAN_RE.sub(" ", text)

    years = sorted(set(_YEAR_RE.findall(text)))
    months = sorted({int(m) for m in _MONTH_RE.findall(rest)})

    values = []
    for enum in enums or []:
        for value in enum.get("values") or []:
            value = str(value)
            if value and value in text:
                values.append({
                    "column": enum.get("column", ""),
                    "label": enum.get("label") or enum.get("column") or "取值",
                    "value": value,
                })
    return {"years": years, "quarters": extract_quarters(text),
            "months": months, "spans": spans, "values": values}


def _month_present(month: int, body: str) -> bool:
    """月份是否还在语句里。

    容忍各种写法：'2025-03'、'2025-3'、'03-'、"'03'"、'3月'。
    宁可放宽（漏判），也不要因为写法差异把正常结果拦掉。
    """
    tokens = (f"-{month:02d}", f"-{month}", f"{month:02d}-", f"'{month:02d}'",
              f"{month}月", f"{month:02d}月")
    if any(token in body for token in tokens):
        return True
    # 还有一些查询不写日期字符串，直接比月份数字：
    # `BETWEEN 4 AND 6`、`month = 6`、`IN (4,5,6)`。
    # 这种情况只在比较/区间的上下文里认，并且先剔掉 LIMIT ——
    # 否则 `LIMIT 6` 里的 6 会被当成"六月还在"，把缩了范围的查询放过去。
    bare = _LIMIT_RE.sub(" ", body)
    return re.search(_BARE_MONTH_CTX % month, bare, re.IGNORECASE) is not None


def _span_present(start: int, end: int, body: str) -> bool:
    """一段月份区间是否还在语句里。

    两种等价写法都要放行，否则会把正确结果拦掉：
    - 逐月展开：IN ('2025-03','2025-04','2025-05')
    - 日期区间：>= '2025-03-01' AND < '2025-06-01'（上界写成下一个月 1 日，
      这时区间末月根本不会以「-05」出现在语句里）
    """
    if not _month_present(start, body):
        return False
    return _month_present(end, body) or _month_present(end % 12 + 1, body)


def _quarter_present(year: Optional[int], quarter: int, body: str) -> bool:
    """这一季度是否还在语句里。

    写法不唯一，三种都认（多认几种是故意的，拦错人比漏判更伤）：
    - 字面：'2025Q2'、'2025-Q2'、'Q2'、'第二季度'
    - 逐月展开：IN ('2025-04','2025-05','2025-06')
    - 日期区间：>= '2025-04-01' AND < '2025-07-01'（上界是下季度首月 1 日）
    只问季度、不带年份时（「Q2 比 Q1」）不追究年份。
    """
    for match in _QUARTER_RE.finditer(body):
        if _quarter_of(match) != quarter:
            continue
        literal_year = match.group("y")
        if year is None or literal_year is None or int(literal_year) == year:
            return True
    return _span_present(quarter * 3 - 2, quarter * 3, body)


def check_scope(question: str, statement: str, enums: Optional[list] = None) -> List[str]:
    """检查重试语句是否还在回答"用户问的那个范围"。

    返回问题清单（空列表表示没动过）。每条都写成能直接展示给用户的话 ——
    它会被原样放进最终回答里解释"为什么没给数"。
    """
    anchors = extract_anchors(question, enums)
    body = strip_code_fence(statement)
    problems: List[str] = []

    for year in anchors["years"]:
        if year not in body:
            problems.append(f"你问的是 {year} 年，重试的查询里已经找不到 {year}（等于换成了别的年份）")
    for year, quarter in anchors["quarters"]:
        if not _quarter_present(year, quarter, body):
            label = f"{year}年Q{quarter}" if year else f"Q{quarter}"
            problems.append(f"你问的是 {label}（{quarter * 3 - 2}-{quarter * 3} 月），"
                            f"重试的查询里已经找不到这一季度（被换成了别的时段）")
    for month in anchors["months"]:
        if not _month_present(month, body):
            problems.append(f"你问的是 {month} 月，重试的查询里已经找不到 {month} 月")
    for start, end in anchors["spans"]:
        if not _span_present(start, end, body):
            problems.append(f"你问的是 {start}-{end} 月这一段，重试的查询里已经找不到完整区间"
                            f"（被缩成了其中一部分月份）")
    for item in anchors["values"]:
        if item["value"] not in body:
            problems.append(
                f"你限定的{item['label']}「{item['value']}」在重试的查询里被去掉了")
    return problems
