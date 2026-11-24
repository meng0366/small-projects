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
重试生成的语句里，用户原话中的年份、月份、枚举取值必须仍然在。
"""
from __future__ import annotations

import re
from typing import List, Optional

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_MONTH_RE = re.compile(r"(?<!\d)(1[0-2]|0?[1-9])\s*月")
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


def extract_anchors(question: str, enums: Optional[list] = None) -> dict:
    """抽出用户原话里"不能动"的条件：年份、月份、枚举取值。"""
    text = question or ""
    years = sorted(set(_YEAR_RE.findall(text)))
    months = sorted({int(m) for m in _MONTH_RE.findall(text)})

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
    return {"years": years, "months": months, "values": values}


def _month_present(month: int, body: str) -> bool:
    """月份是否还在语句里。

    容忍各种写法：'2025-03'、'2025-3'、'03-'、"'03'"、'3月'。
    宁可放宽（漏判），也不要因为写法差异把正常结果拦掉。
    """
    tokens = (f"-{month:02d}", f"-{month}", f"{month:02d}-", f"'{month:02d}'",
              f"{month}月", f"{month:02d}月")
    return any(token in body for token in tokens)


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
    for month in anchors["months"]:
        if not _month_present(month, body):
            problems.append(f"你问的是 {month} 月，重试的查询里已经找不到 {month} 月")
    for item in anchors["values"]:
        if item["value"] not in body:
            problems.append(
                f"你限定的{item['label']}「{item['value']}」在重试的查询里被去掉了")
    return problems
