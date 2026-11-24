"""取值校验：用户问到的枚举值库里没有时，先澄清，而不是硬映射（P0-1b）。

为什么需要这层（真实试跑依据）：
`schema` 里早就列出了合法取值，还写着"必须严格使用以下字面量，不要用近义词替换"，
模型照样把「办公用品」映射成「配件」、把「电脑」映射成「笔记本」，并给出金额
（395,283.75 / 7,524,472.00）。说明问题不是"不知道取值"，而是"知道对不上就自己挑了个最近的"。
提示词堵不住这种替换，只能用确定性规则拦在生成之前。

规则（宁可多问一句，也不给错数）：
1. 问题里已出现该列的某个合法取值 → 这列放行（用户没说错）；
2. 否则找"取值线索词"（catalog 里配的 hints：品类/分类/类、地区/区域、客户等级…），
   取线索词前面紧邻的中文/字母数字串作为候选值；
3. 候选值含虚词（的、了、是、多少…）、是纯数字、长度不合理 → 判定不是取值，跳过；
4. 候选值不在合法取值里 → 返回澄清信息（带真实可选值）。

只报"能明确看出是在说取值"的情况：漏报最多是维持现状，误报却会让正常问题
被拦下来问一句废话，所以判定从严。
"""
from typing import Optional

STOP_CHARS = set("的了是在有和与及每个各这那我你他她它们多少几哪些什么怎么为什么如何按根据照"
                 "属于或者对把被让给就要会能可以想请帮忙查算看说关于由于从到向跟同也都很更最再"
                 "又只凡全总所")
"""候选值里出现任一虚词，就认为它不是"取值"而是半句话。"""

MIN_CANDIDATE_LEN = 2
MAX_CANDIDATE_LEN = 6


def _candidate_before(question: str, end: int) -> str:
    """取 end 位置前紧邻的一段中文/字母数字（最长 MAX_CANDIDATE_LEN 个字符）。

    「办公用品类产品」+ 线索词「类」→ 得到「办公用品」；
    「帮我看看家用电器类」+ 线索词「类」→ 遇到虚词「看」就断开，得到「家用电器」；
    「最高的品类」+ 线索词「品类」→ 紧邻就是「的」，直接断开得到空串，被长度规则否掉。
    """
    window = question[max(0, end - MAX_CANDIDATE_LEN):end]
    chars = []
    for ch in reversed(window):
        if not (ch.isalnum() or "\u4e00" <= ch <= "\u9fff"):
            break
        if ch in STOP_CHARS:   # 虚词是边界，不并进候选值
            break
        chars.append(ch)
    return "".join(reversed(chars))


def _is_fixed_phrase(candidate: str, hint: str, hints: list) -> bool:
    """候选值 + 线索词 是否本身就是列名（如「客户」+「等级」=「客户等级」）。

    这种情况说明线索词只是固定搭配的一部分，不是用户在指一个取值，
    否则"各客户等级的销售额"会被误判成"库里没有客户这个等级"。
    """
    joined = candidate + hint
    return any(joined == str(h) for h in hints)


def _looks_like_value(candidate: str, values: list) -> bool:
    """候选词是否像一个"取值"（而不是虚词、数字或已存在的合法值）。"""
    if not (MIN_CANDIDATE_LEN <= len(candidate) <= MAX_CANDIDATE_LEN):
        return False
    if candidate in values:
        return False
    if candidate.isdigit():
        return False
    return not any(ch in STOP_CHARS for ch in candidate)


def find_unknown_enum(question: str, enums: list) -> Optional[dict]:
    """找出"用户点名了某个枚举列、但给的取值库里没有"的情况。

    返回 {table, column, label, candidate, values}；没命中返回 None。
    """
    question = (question or "").strip()
    if not question:
        return None

    for enum in enums or []:
        values = [str(v) for v in (enum.get("values") or [])]
        if not values:
            continue
        # 用户说的值本来就在库里 → 这列没问题
        if any(v and v in question for v in values):
            continue

        # 同一句话可能有多个线索词命中且边界不同：
        # 「办公用品类」里既有「品类」（取到「办公用」）又有「类」（取到「办公用品」），
        # 上下文更长的那种解读更接近用户想说的取值，所以取最长候选。
        best = None
        hints = [str(h) for h in (enum.get("hints") or []) if h]
        for hint in hints:
            start = 0
            while True:
                idx = question.find(hint, start)
                if idx < 0:
                    break
                start = idx + 1
                candidate = _candidate_before(question, idx)
                if not _looks_like_value(candidate, values):
                    continue
                if _is_fixed_phrase(candidate, hint, hints):
                    continue
                if best is None or len(candidate) > len(best):
                    best = candidate

        if best:
            return {
                "table": enum.get("table", ""),
                "column": enum.get("column", ""),
                "label": enum.get("label") or enum.get("column", ""),
                "candidate": best,
                "values": values,
            }
    return None


def build_clarify_answer(flag: dict) -> str:
    """澄清文案：直接说"没有这个值"，并列出真实可选值。"""
    values = [str(v) for v in (flag.get("values") or [])]
    label = flag.get("label") or flag.get("column") or "取值"
    candidate = flag.get("candidate", "")
    example = values[0] if values else ""

    parts = [
        f"你提到的「{candidate}」不在库里的{label}取值中。",
        f"现有可选值：{' / '.join(values)}",
        "",
        "我没有拿近似的值去顶替 —— 那样即使给出数字，口径也是错的。",
    ]
    if example:
        parts.append(f"如果指的是其中某一项，可以直接问「{example}的销售额是多少」；"
                     f"也可以告诉我不用这个筛选条件。")
    return "\n".join(parts)
