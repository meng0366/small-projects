"""追问建议（阶段 6 新功能）。

借鉴点：OpenChatBI 的"交互式引导"与 natural-sql 的 query suggestions——
答完一个问题后给出可以接着问的方向，降低"不知道该问什么"的门槛。

**为什么用规则而不是让模型生成**：
- 追问只在答完之后展示，若再调一次模型，每题都多一次成本，收益却很小；
- 规则可离线测试、结果稳定，不会出现"推荐的维度表里没有"这种硬编错误；
- 规则表就下面几行，业务换数据时改这一个 dict 即可。

规则只从**结果集真实存在的列**和**问题里没涉及的业务维度**出发，
不猜表结构，因此不会推荐不存在的字段。
"""
from typing import List, Optional

import pandas as pd

# 业务维度：关键词 → 展示名。命中结果列或问题时都按"已涉及"处理。
DIMENSION_KEYWORDS = {
    "区域": "区域",
    "地区": "区域",
    "region": "区域",
    "品类": "品类",
    "类别": "品类",
    "category": "品类",
    "客户等级": "客户等级",
    "等级": "客户等级",
    "level": "客户等级",
    "客户": "客户",
    "customer": "客户",
    "商品": "商品",
    "产品": "商品",
    "product": "商品",
}

DIMENSION_QUESTIONS = {
    "区域": "各区域的销售额是多少？",
    "品类": "各品类的销售额是多少？",
    "客户等级": "各客户等级的销售额是多少？",
    "客户": "销售额最高的前 5 个客户是谁？",
    "商品": "销售额最高的前 5 个商品是什么？",
}

TIME_KEYWORDS = ("月", "季度", "日期", "时间", "趋势", "走势", "month", "date", "trend")
TIME_QUESTION = "每个月的销售额走势如何？"

MAX_VALUE_LEN = 20


def _columns_text(df: Optional[pd.DataFrame]) -> str:
    try:
        return " ".join(str(c).lower() for c in df.columns)
    except Exception:
        return ""


def _covered_dimensions(text: str) -> set:
    """文本里已经出现过的业务维度。"""
    text = (text or "").lower()
    return {label for keyword, label in DIMENSION_KEYWORDS.items() if keyword in text}


def _first_value(df: Optional[pd.DataFrame]) -> Optional[str]:
    """取结果第一行第一列的文本值（用于"只看 X"这类收窄建议）。"""
    if not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] == 0:
        return None
    value = df.iloc[0, 0]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return None    # 数值型首列（多半是指标本身）不适合做收窄建议
    text = str(value).strip()
    if not text or len(text) > MAX_VALUE_LEN:
        return None
    return text


def suggest_followups(question: str, df: Optional[pd.DataFrame],
                      max_suggestions: int = 3) -> List[str]:
    """按结果集与问题给出最多 `max_suggestions` 条追问建议。"""
    if max_suggestions <= 0:
        return []

    question = (question or "").strip()
    columns_text = _columns_text(df)
    covered = _covered_dimensions(question) | _covered_dimensions(columns_text)

    candidates: List[str] = []

    # 规则 1：结果就是一个具体对象的明细/指标 → 建议收窄到其中最值得看的那一项
    # （如果问题里已经限定到这一项了，再推荐"只看它"就是废话）
    first = _first_value(df)
    if first and first not in question:
        candidates.append(f"只看「{first}」的情况如何？")

    # 规则 2/3：先给一条"换维度对比"，再给一条"看趋势"，避免三格全被同类建议占满
    dim_candidates = [DIMENSION_QUESTIONS[label]
                      for label in ("区域", "品类", "客户等级", "客户", "商品")
                      if label not in covered]
    if dim_candidates:
        candidates.append(dim_candidates.pop(0))
    if not any(keyword in question.lower() or keyword in columns_text for keyword in TIME_KEYWORDS):
        candidates.append(TIME_QUESTION)
    candidates.extend(dim_candidates)

    # 去重、去空、去掉和原问题一样的
    result: List[str] = []
    for item in candidates:
        if not item or item == question or item in result:
            continue
        result.append(item)
        if len(result) >= max_suggestions:
            break
    return result
