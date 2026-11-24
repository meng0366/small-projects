"""评测判分（对应 S5.1 / S5.2）。

设计取舍：为什么不让大模型打分？
- 大模型打分主观、不可复现、还要额外花钱，也没法解释"为什么错"；
- 本项目的评测问题都有唯一数值答案，用**金标准 SQL 在同一个库上执行一遍**就是标准答案；
- 判分只比"值"，不比列名与列顺序——模型把列叫「销售额」还是「total_amount」不影响对错。

三种判定模式：
- scalar：单值汇总题（总额、占比、客单价…），要求结果恰好 1 行，且期望值出现在该行；
- text：答案是某个维度值（哪个区域最高、哪个月最好），取结果第一行判断是否包含该值；
- table：多行结果（分组汇总、Top-N），比对行集合与每一列的值。

数值比较用容差而不是四舍五入：金额容差 0.011（覆盖两位小数的浮点误差），
比率类题目可以在用例里把 atol 调小到 1e-4。
"""
from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_ATOL = 0.011
# 结果里常见的"装饰"：货币符号、千分位、单位后缀，比对前统一去掉
_CURRENCY_RE = re.compile(r"^[¥$￥]\s*")
_UNIT_RE = re.compile(r"\s*(元|件|个|笔|人|次|条)$")


@dataclass
class JudgeResult:
    """一次判分的结果，detail 用于失败复盘。"""
    matched: bool
    mode: str = ""          # scalar / text / table / strict / lenient
    detail: str = ""


# ---------------------------------------------------------------- 值规范化

def norm_cell(value: Any) -> Any:
    """把单元格转成可比较的值：数值统一成 float，日期转文本，文本去空格与装饰符。"""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime, pd.Timestamp)):
        return str(value)[:10]

    text = str(value).strip()
    if not text:
        return None
    text = _CURRENCY_RE.sub("", text)
    text = _UNIT_RE.sub("", text)
    text = text.replace(",", "").replace("，", "").strip()
    try:
        return float(text)
    except ValueError:
        return text


def _num_equal(a: float, b: float, atol: float) -> bool:
    """数值相等判断：绝对容差 + 极小相对容差（防止大数尾差）。"""
    return abs(a - b) <= max(atol, 1e-9 * max(abs(a), abs(b)))


def _cell_equal(a: Any, b: Any, atol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return _num_equal(a, b, atol)
    return str(a) == str(b)


def _sort_key(v: Any) -> Tuple[int, Any]:
    """混合类型排序：数值在前按大小，文本在后按字典序，空值最后。"""
    if v is None:
        return (2, "")
    if isinstance(v, float):
        return (0, v)
    return (1, str(v))


# ---------------------------------------------------------------- 结构转换

def as_frame(result: Any) -> Optional[pd.DataFrame]:
    """把 Agent 结果统一成 DataFrame；标量与 Series 也兼容。"""
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame().reset_index()
    if isinstance(result, (list, tuple, dict)):
        try:
            return pd.DataFrame(result)
        except Exception:
            return None
    return pd.DataFrame({"结果": [result]})


def to_matrix(frame: pd.DataFrame) -> List[List[Any]]:
    """DataFrame → 规范化二维列表（保持行列顺序）。"""
    return [[norm_cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]


def _columns(matrix: List[List[Any]]) -> List[List[Any]]:
    """按列切分矩阵。"""
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]


# ---------------------------------------------------------------- 列/行比较

def _col_equal(gold: Sequence[Any], got: Sequence[Any], atol: float) -> bool:
    """两列是否"值一致"：忽略顺序，逐元素按容差比较。"""
    if len(gold) != len(got):
        return False
    g = sorted(gold, key=_sort_key)
    r = sorted(got, key=_sort_key)
    return all(_cell_equal(a, b, atol) for a, b in zip(g, r))


def _col_equal_any_scale(gold: Sequence[Any], got: Sequence[Any],
                         atol: float) -> Tuple[bool, int]:
    """列比较，容忍量纲差异（占比写成 0.2345 还是 23.45）。

    返回 (是否匹配, 使用的倍率)。倍率仅用于说明，不参与后续计算。
    """
    if _col_equal(gold, got, atol):
        return True, 1
    for factor in (100.0, 0.01):
        scaled = []
        for v in got:
            scaled.append(v * factor if isinstance(v, float) else v)
        if _col_equal(gold, scaled, atol):
            return True, int(factor) if factor >= 1 else 1  # 0.01 不单独标注
    return False, 1


def _row_key(row: Sequence[Any]) -> Tuple:
    return tuple(row)


def _rows_equal(gold: List[List[Any]], got: List[List[Any]],
                atol: float) -> bool:
    """行多重集比较：仍然逐格比较（不排序后再比，避免列内重复值掩盖顺序错位）。"""
    if len(gold) != len(got):
        return False
    if all(len(g) == len(r) for g, r in zip(gold, got)):
        if all(_cell_equal(a, b, atol) for grow, rrow in zip(gold, got)
               for a, b in zip(grow, rrow)):
            return True
    # 两边的行顺序可能不同（模型按不同列排序），用配对搜索而不是直接 zip
    remaining = list(got)
    for grow in gold:
        for i, rrow in enumerate(remaining):
            if len(rrow) == len(grow) and all(
                    _cell_equal(a, b, atol) for a, b in zip(grow, rrow)):
                remaining.pop(i)
                break
        else:
            return False
    return not remaining


# ---------------------------------------------------------------- 主入口

def judge(result: Any, golds: List[pd.DataFrame], spec: dict) -> JudgeResult:
    """按用例声明的 mode 判分；golds 为「标准答案 + 等价口径备选」，命中任一即通过。"""
    mode = spec.get("mode", "table")
    atol = float(spec.get("atol", DEFAULT_ATOL))

    frame = as_frame(result)
    if frame is None or frame.empty:
        return JudgeResult(False, mode, "没有返回结果（查询失败或结果为空）")

    matrix = to_matrix(frame)
    if not matrix or not matrix[0]:
        return JudgeResult(False, mode, "结果为空表")

    if mode == "scalar":
        if not spec.get("allow_multi_row") and len(matrix) != 1:
            return JudgeResult(False, mode,
                               f"期望 1 行汇总结果，实际返回 {len(matrix)} 行")
        for gold in golds:
            target = to_matrix(gold)[0][0]
            if any(_cell_equal(c, target, atol) for c in matrix[0]):
                return JudgeResult(True, mode, "汇总值与标准答案一致")
        target = to_matrix(golds[0])[0][0]
        return JudgeResult(False, mode,
                           f"期望 {target}，实际返回 {_fmt_row(matrix[0])}")

    if mode == "text":
        # 取第一行：模型若返回降序排行，第一行就是它认为的答案
        first = matrix[0]
        for gold in golds:
            target = to_matrix(gold)[0][0]
            target_text = str(target)
            if any(str(c) is not None and target_text in str(c) for c in first if c is not None):
                return JudgeResult(True, mode, f"答案包含「{target_text}」")
        target = str(to_matrix(golds[0])[0][0])
        return JudgeResult(False, mode,
                           f"期望「{target}」，实际首行是 {_fmt_row(first)}")

    # table
    last: Optional[JudgeResult] = None
    for gold in golds:
        gm = to_matrix(gold)
        res = _judge_table(gm, matrix, atol, spec)
        if res.matched:
            return res
        last = res
    detail = last.detail if last else "结果不一致"
    detail += f"；期望首行 {_fmt_row(to_matrix(golds[0])[0])}，实际首行 {_fmt_row(matrix[0])}"
    return JudgeResult(False, "table", detail)


def _judge_table(gold: List[List[Any]], got: List[List[Any]], atol: float,
                 spec: dict) -> JudgeResult:
    allow_extra_cols = bool(spec.get("allow_extra_cols"))
    allow_extra_rows = bool(spec.get("allow_extra_rows"))

    if len(gold) != len(got):
        if not allow_extra_rows or len(got) < len(gold):
            return JudgeResult(False, "table",
                               f"行数不一致：期望 {len(gold)} 行，实际 {len(got)} 行")
    if not allow_extra_cols and len(gold[0]) != len(got[0]):
        return JudgeResult(False, "table",
                           f"列数不一致：期望 {len(gold[0])} 列，实际 {len(got[0])} 列")

    if _rows_equal(gold, got, atol):
        return JudgeResult(True, "strict", "逐行逐列一致")

    # 列名与列顺序不敏感：给标准答案的每一列在结果里找一列"值一致"的
    gold_cols = _columns(gold)
    got_cols = _columns(got)
    used = set()
    scaled = []
    for gi, gcol in enumerate(gold_cols):
        hit = None
        for ri, rcol in enumerate(got_cols):
            if ri in used:
                continue
            ok, factor = _col_equal_any_scale(gcol, rcol, atol)
            if ok:
                hit = (ri, factor)
                break
        if hit is None:
            return JudgeResult(False, "table",
                               f"第 {gi + 1} 列的值与期望不符（期望含 {_fmt_row(gcol[:5])}…）")
        used.add(hit[0])
        if hit[1] != 1:
            scaled.append(hit[0] + 1)
    if not allow_extra_cols and len(used) != len(got_cols):
        return JudgeResult(False, "table", "存在多余列")
    note = "列内容一致（列名或列顺序不同）"
    if scaled:
        note += f"；第 {scaled} 列按百分比换算后一致"
    return JudgeResult(True, "lenient", note)


def _fmt_row(row: Sequence[Any], limit: int = 4) -> str:
    """把一行格式化成便于阅读的短文本。"""
    parts = []
    for v in list(row)[:limit]:
        if isinstance(v, float):
            parts.append(f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}")
        else:
            parts.append(str(v))
    if len(row) > limit:
        parts.append("…")
    return "(" + ", ".join(parts) + ")"
