"""指标归因（P1-2）：把「为什么这个指标变了」拆成「哪些分项贡献了多少」。

问题形态：
    - 时段对比：「为什么 2025 年 6 月的销售额比 5 月高？」
    - 维度对比：「为什么华东的销售额比华北高？」
    两者本质相同：一个**对比列**（月份 / 区域）+ 一个**拆解维度**（品类 / 客户等级）。

做法（职责划分是本模块最要紧的设计）：
    模型只负责出一条"对比列 × 拆解维度 × 指标"的交叉表，
    差分、占比、排序、闭合校验**全部由这里的纯函数算**。
    让模型"读表算占比"必然会出现三项占比加不到 100% 这种看不见的错，
    而这类错一旦进了结论，读的人无从核对。

三条硬约束（都是本项目反复踩坑换来的）：
1. **算数不进模型**：数字全由代码产出，可单元测试、可复算；
2. **拆不出来就说拆不出来**：结果里没有可用分类列时返回 ok=False，
   由上层降级成普通明细，绝不硬凑一个漂亮结论；
3. **比率指标不可加总**：客单价/单价/率这类指标，分项变化相加没有业务含义，
   只按变化幅度排序，并在结论里标明，避免"分项加起来对不上"的误导。
"""
import calendar
import re
from typing import Optional

import pandas as pd

# 出现这些词，才认为用户在问"为什么变了"（而不是单纯要一个数）
CAUSE_WORDS = ("为什么", "为啥", "原因", "归因", "导致", "拉动", "拉低", "拉高", "贡献")

# 对比句式的方向标记：出现"到/至"表示 前→后（前为基期）；出现"比/较"表示 前为被解释对象
TO_MARKERS = ("到", "至", "→", "->")
THAN_MARKERS = ("比", "较", "高于", "低于", "超过", "不如", "vs", "VS")

# 只说到一个时段时，允许默认与上一期比较；但必须是明确的涨跌/变化词，避免过度解读
SOLO_CHANGE_WORDS = ("下降", "下滑", "减少", "上升", "增长", "增加", "提升", "降低",
                     "下跌", "上涨", "波动", "变化", "高", "低")

# 问题的拆解维度线索词 → 建议维度名
DIMENSION_HINTS = {
    "客户等级": "客户等级", "等级": "客户等级", "级别": "客户等级",
    "品类": "品类", "分类": "品类", "类目": "品类",
    "区域": "区域", "地区": "区域", "大区": "区域",
    "商品": "商品", "产品": "商品",
}

# 比率类指标：分项变化不可加总
RATIO_HINTS = ("率", "占比", "比例", "客单价", "单价", "均价", "平均", "均值", "百分比",
               "share", "ratio", "pct", "avg", "mean")

# 结果里可能出现的合计行，做归因时必须剔除，否则分项翻倍
TOTAL_LABELS = ("合计", "总计", "全部", "汇总", "total", "all")

MAX_DIM_CARDINALITY = 30
"""拆解维度的最大基数：超过这个数（订单号、客户名等）拆开没有业务意义，也读不下去。"""

MIN_DIM_CARDINALITY = 2

ISO_PERIOD_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})(?![\d-])")
CN_PERIOD_RE = re.compile(
    r"(?:(?P<y>\d{4})\s*[-/年\.]\s*)?(?P<m>\d{1,2}|[一二三四五六七八九十两]{1,3})\s*月"
)
YEAR_ONLY_RE = re.compile(r"(?<![\d年])(\d{4})\s*年(?!\s*[\d一二三四五六七八九十])")

CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


# ---------------------------------------------------------------- 时段解析
def cn_to_int(text: str) -> Optional[int]:
    """把「6」/「六」/「十二」这类月份写法转成整数；转不了返回 None。"""
    text = str(text).strip()
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if text.startswith("十"):          # 十一、十二
        rest = text[1:]
        return 10 + CN_DIGITS[rest] if rest in CN_DIGITS else None
    if text.endswith("十") and text[0] in CN_DIGITS:   # 二十（本项目用不到，稳妥处理）
        return CN_DIGITS[text[0]] * 10
    return CN_DIGITS.get(text)


def period_key(text) -> Optional[tuple]:
    """把各种时段写法归一成 (年, 月)，缺失的部分为 None。

    '2025-06' / '2025/6' / '2025年6月' / '六月' / '202506' / '2025年' 都能识别。
    归一化的目的：让用户口径（'6月'）与库里口径（'2025-06'）能对上，
    而不是要求用户按数据库格式提问。
    """
    s = str(text).strip()
    if not s:
        return None

    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})(?:[-/.]\d{1,2})?", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{4})\s*年\s*(\d{1,2})\s*月?", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{4})(\d{2})", s)          # 202506
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{1,2}|[一二三四五六七八九十两]{1,3})\s*月", s)
    if m:
        month = cn_to_int(m.group(1))
        if month and 1 <= month <= 12:
            return None, month
    m = re.fullmatch(r"(\d{4})\s*年", s)
    if m:
        return int(m.group(1)), None
    return None


def _compatible(a: tuple, b: tuple) -> bool:
    """两个 (年, 月) 是否指同一时段；只问到月份时忽略年份。"""
    y1, m1 = a
    y2, m2 = b
    if m1 is not None and m2 is not None and m1 != m2:
        return False
    if y1 is not None and y2 is not None and y1 != y2:
        return False
    return True


def _find_periods(text: str) -> list:
    """找出问题里出现的时段，按出现位置排序（重叠的取更长的那段）。"""
    hits = []
    for m in ISO_PERIOD_RE.finditer(text):
        hits.append({"start": m.start(), "end": m.end(), "raw": m.group(0),
                     "key": (int(m.group(1)), int(m.group(2)))})
    for m in CN_PERIOD_RE.finditer(text):
        month = cn_to_int(m.group("m"))
        if not month or not 1 <= month <= 12:
            continue
        year = int(m.group("y")) if m.group("y") else None
        hits.append({"start": m.start(), "end": m.end(), "raw": m.group(0).strip(),
                     "key": (year, month)})
    for m in YEAR_ONLY_RE.finditer(text):
        hits.append({"start": m.start(), "end": m.end(), "raw": m.group(0),
                     "key": (int(m.group(1)), None)})

    hits.sort(key=lambda h: (h["start"], -(h["end"] - h["start"])))
    picked, cursor = [], -1
    for hit in hits:
        if hit["start"] < cursor:        # 与已选区间重叠 → 丢弃短的
            continue
        hit["raw"] = re.sub(r"\s+", "", hit["raw"])
        picked.append(hit)
        cursor = hit["end"]
    return picked


def _dimension_hints(question: str) -> list:
    """从问题里读出用户想按什么拆，**按出现顺序**返回去重后的维度名。

    顺序很重要：说"按品类和客户等级拆"时，品类是一级、客户等级是二级；
    原来的实现按字典插入顺序取第一个命中，用户先说"客户等级"也会被排成品类。
    同时点两个维度就是二级拆解（P2-3），只点一个就只做一级。
    """
    hits = []
    for word, dim in DIMENSION_HINTS.items():
        start = question.find(word)
        if start >= 0:
            hits.append((start, dim))
    hits.sort()
    dims = []
    for _, dim in hits:
        if dim not in dims:
            dims.append(dim)
    return dims


def _previous_period(key: tuple) -> Optional[tuple]:
    """上一期（月粒度）；月份缺失时无法推，返回 None。"""
    year, month = key
    if month is None:
        return None
    if month == 1:
        return (year - 1 if year else None), 12
    return year, month - 1


# ---------------------------------------------------------------- 意图识别
def detect_attribution(question: str, enums: list = None) -> Optional[dict]:
    """判断是不是归因问题，并抽出「谁比谁」。

    返回 None 表示"不是归因"，交给普通问答路径 —— 宁可漏判，
    也不要把普通问题硬当归因（那会逼模型输出一张没人要的交叉表）。
    """
    q = (question or "").strip()
    if not q or len(q) > 300:
        return None
    if not any(w in q for w in CAUSE_WORDS):
        return None

    hints = _dimension_hints(q)
    periods = _find_periods(q)
    if len(periods) >= 2:
        first, second = periods[0], periods[1]
        between = q[first["end"]:second["start"]]
        if any(m in between for m in TO_MARKERS):
            base, current = first, second            # 「5 月到 6 月」
        elif any(m in between for m in THAN_MARKERS):
            base, current = second, first            # 「6 月比 5 月」
        else:
            # 无方向标记：先说到的通常是被解释对象（「6 月的销售额为什么比上月高」）
            base, current = second, first
        return {
            "kind": "period",
            "base": base["raw"], "current": current["raw"],
            "base_key": base["key"], "current_key": current["key"],
            "assumed": False,
            "dimension_hints": hints,
            "dimension_hint": (hints or [""])[0],
        }

    if len(periods) == 1 and any(w in q for w in SOLO_CHANGE_WORDS):
        # 只给了一个时段：默认与上一期比。这个假设必须在结论里写明，
        # 否则用户会以为系统比的是他心里那个基准。
        current = periods[0]
        prev = _previous_period(current["key"])
        if prev:
            return {
                "kind": "period",
                "base": f"{prev[0]}年{prev[1]}月" if prev[0] else f"{prev[1]}月",
                "current": current["raw"],
                "base_key": prev, "current_key": current["key"],
                "assumed": True,
                "dimension_hints": hints,
                "dimension_hint": (hints or [""])[0],
            }

    pair = _find_enum_pair(q, enums or [])
    if pair:
        return pair
    return None


def _find_enum_pair(question: str, enums: list) -> Optional[dict]:
    """维度对比：问题里同时出现某个枚举列的两个取值（华东 / 华北）。"""
    for group in enums:
        values = [str(v) for v in (group.get("values") or []) if v and str(v) in question]
        if len(values) < 2:
            continue
        values.sort(key=question.index)
        first, second = values[0], values[1]
        between = question[question.index(first) + len(first): question.index(second)]
        if any(m in between for m in TO_MARKERS):
            base, current = first, second
        else:
            base, current = second, first            # 「华东比华北高」→ 解释华东
        return {
            "kind": "dimension",
            "base": base, "current": current,
            "base_key": None, "current_key": None,
            "assumed": False,
            "column": group.get("column", ""),
            "dimension_hints": _dimension_hints(question),
            "dimension_hint": (_dimension_hints(question) or [""])[0],
        }
    return None


# ---------------------------------------------------------------- 结果拆解
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _value_matches(value, label) -> bool:
    """数据里的某个取值是否对应用户说的那个对象（时段按年月归一，其余按字面包含）。"""
    vk, lk = period_key(value), period_key(label)
    if vk and lk:
        return _compatible(vk, lk)
    v, l = str(value).strip(), str(label).strip()
    if not v or not l:
        return False
    return v == l or l in v or v in l


def _limit_year(frame: pd.DataFrame, col, mask: pd.Series, label: str):
    """用户没写年份、数据又跨多年时，只取最新一年，并把这件事说出来。

    不处理的后果很具体：库里同时有 2024 与 2025 时，「6 月」会把两年的 6 月
    加在一起，得出一个既不是今年也不是去年的数，而且看不出错。
    """
    key = period_key(label)
    if not key or key[0] is not None:
        return mask, ""
    keys = frame.loc[mask, col].map(period_key)
    years = sorted({k[0] for k in keys if k and k[0]})
    if len(years) <= 1:
        return mask, ""
    latest = years[-1]
    keep = mask & frame[col].map(
        lambda v: (period_key(v) or (None, None))[0] == latest)
    return keep, f"问题里没写年份，已按数据中最新的 {latest} 年计算"


def _find_compare_column(frame: pd.DataFrame, intent: dict):
    """找出「对比列」以及归入基期/本期的行掩码，返回 (列名, side, 说明)。"""
    base, current = intent.get("base", ""), intent.get("current", "")
    best = None
    for col in frame.columns:
        if _is_numeric(frame[col]):
            continue
        values = frame[col].dropna()
        if values.empty:
            continue
        base_mask = values.map(lambda v: _value_matches(v, base))
        curr_mask = values.map(lambda v: _value_matches(v, current))
        if not base_mask.any() or not curr_mask.any():
            continue
        # 两边都命中同一批行说明这一列区分不出对比双方（例如列值全一样）
        if bool((base_mask & curr_mask).all()):
            continue
        distinct = values[base_mask | curr_mask].astype(str).nunique()
        score = distinct + (2 if intent.get("column") and intent["column"] == str(col) else 0)
        if best is None or score > best[0]:
            best = (score, col)
    if not best:
        return None, None, ""

    col = best[1]
    base_mask = frame[col].map(lambda v: _value_matches(v, base)).fillna(False)
    curr_mask = frame[col].map(lambda v: _value_matches(v, current)).fillna(False)
    notes = []
    if intent.get("kind") == "period":
        base_mask, note = _limit_year(frame, col, base_mask, base)
        if note:
            notes.append(note)
        curr_mask, note = _limit_year(frame, col, curr_mask, current)
        if note and note not in notes:
            notes.append(note)

    side = pd.Series(index=frame.index, dtype=object)
    side[base_mask] = "base"
    side[curr_mask] = "current"
    return col, side, "；".join(notes)


def _is_total_label(value) -> bool:
    text = str(value).strip().lower()
    return any(t in text for t in TOTAL_LABELS)


def _pick_metric(frame: pd.DataFrame, question: str, exclude: set) -> Optional[str]:
    """选指标列：优先与问题里提到的指标名对得上，否则取第一个数值列。"""
    numeric = [c for c in frame.columns if c not in exclude and _is_numeric(frame[c])]
    if not numeric:
        return None
    if len(numeric) > 1:
        for col in numeric:
            name = str(col)
            for word in ("销售额", "订单量", "销量", "客单价", "单价", "折扣", "数量", "金额"):
                if word in question and (word in name or name in word):
                    return col
    return numeric[0]


def _pick_dimensions(frame: pd.DataFrame, exclude: set, hints=()) -> tuple:
    """选一级与二级拆解维度：优先用户点名的，否则一级取基数最大的可用列。

    只按用户点名给二级维度、不自动猜：猜错会产出一段"看着很深入其实跑偏"的结论，
    比只给一级更糟。用户点了两个维度但结果里只有一层时，`decompose` 会说明原因。
    """
    candidates = []
    for col in frame.columns:
        if col in exclude or _is_numeric(frame[col]):
            continue
        nunique = frame[col].dropna().astype(str).nunique()
        if MIN_DIM_CARDINALITY <= nunique <= MAX_DIM_CARDINALITY:
            candidates.append((str(col), nunique))
    if not candidates:
        return None, None

    names = [name for name, _ in sorted(candidates, key=lambda item: -item[1])]
    matched = []
    for hint in hints:
        if not hint:
            continue
        for name in names:
            if name in matched:
                continue
            if hint in name or name in hint:
                matched.append(name)
                break
    if not matched:
        return names[0], None
    return matched[0], (matched[1] if len(matched) > 1 else None)


def _is_ratio_metric(metric: str) -> bool:
    name = str(metric).lower()
    return any(h in name for h in RATIO_HINTS)


def _month_days(key: tuple) -> Optional[int]:
    """该月有多少天；没写年份时按平年（2025）算，避免闰年 2 月把提示吃掉。"""
    year, month = key or (None, None)
    if month is None:
        return None
    return calendar.monthrange(year or 2025, month)[1]


def decompose(frame, intent: dict, question: str = "") -> dict:
    """把交叉表拆成「分项 × 基期 / 本期 / 变化 / 占比」。

    成功返回 ok=True 的字典（含可直接展示的 table），失败返回 ok=False + reason。
    **不做任何猜测性填充**：找不到对比列或维度就老实失败，让上层降级。
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {"ok": False, "reason": "结果为空，没有可拆解的数据"}
    if frame.shape[1] < 2:
        return {"ok": False, "reason": "结果只有一列，没有可拆分的维度"}

    comp_col, side, year_note = _find_compare_column(frame, intent)
    if not comp_col:
        return {"ok": False,
                "reason": f"结果里找不到与「{intent.get('base')}」「{intent.get('current')}」"
                          f"对应的对比列"}

    metric = _pick_metric(frame, question, exclude={comp_col})
    if not metric:
        return {"ok": False, "reason": "结果里没有数值列，无法比较变化"}

    hints = [h for h in (intent.get("dimension_hints")
                         or [intent.get("dimension_hint", "")]) if h]
    dim, dim2 = _pick_dimensions(frame, exclude={comp_col, metric}, hints=hints)
    if not dim:
        return {"ok": False,
                "reason": "结果只有对比双方的总量，没有拆到业务维度（需要一张"
                          "「对比列 × 维度 × 指标」的交叉表才能归因）"}

    sub = frame.loc[side.notna()].copy()
    if sub.empty:
        return {"ok": False, "reason": "没有取到对比双方的明细行"}

    # 合计行会让分项翻倍，必须在拆解前剔除
    sub = sub[~sub[dim].map(_is_total_label)]
    if sub.empty:
        return {"ok": False, "reason": "结果里只有合计行，没有分项明细"}

    sub["__side__"] = side.loc[sub.index]
    sub[dim] = sub[dim].fillna("(未分类)").astype(str)

    pivot = sub.groupby([dim, "__side__"], dropna=False)[metric].sum().unstack("__side__")
    pivot = pivot.reindex(columns=["base", "current"])
    missing_base = [str(i) for i in pivot.index if pd.isna(pivot.loc[i, "base"])]
    missing_current = [str(i) for i in pivot.index if pd.isna(pivot.loc[i, "current"])]
    pivot = pivot.fillna(0.0)

    table = pd.DataFrame({
        "维度": pivot.index.astype(str),
        "基期": pivot["base"].round(2).to_numpy(),
        "本期": pivot["current"].round(2).to_numpy(),
    })
    table["变化"] = (table["本期"] - table["基期"]).round(2)

    base_total = float(table["基期"].sum())
    current_total = float(table["本期"].sum())
    delta = round(current_total - base_total, 2)

    additive = not _is_ratio_metric(metric)
    # 比率类指标的分项差异不可相加，占比没有意义 → 只按变化幅度排序。
    # 分子乘上总变化的方向：总额在跌时，跟着跌的项应显示为正贡献，
    # 否则"占降幅 43%"会被写成"-43%"，读起来像在说反话。
    direction = 1 if delta >= 0 else -1
    if additive and delta != 0:
        table["贡献%"] = (table["变化"] * direction / abs(delta) * 100).round(1)

    table = table.reindex(table["变化"].abs().sort_values(ascending=False).index)
    table = table.reset_index(drop=True)

    # 二级拆解：只在用户点名了第二个维度、且结果里真有这一列时才做。
    # 下钻对象取变化最大的那一项 —— 归因要回答"谁在动"，不是把所有项都拆一遍。
    drill = None
    if dim2 and not table.empty:
        top_item = str(table.iloc[0]["维度"])
        item_delta = float(table.iloc[0]["变化"])
        item_rows = sub[sub[dim] == top_item].copy()
        item_rows[dim2] = item_rows[dim2].fillna("(未分类)").astype(str)
        pivot2 = (item_rows.groupby([dim2, "__side__"], dropna=False)[metric]
                  .sum().unstack("__side__").reindex(columns=["base", "current"]).fillna(0.0))
        drill_table = pd.DataFrame({
            "维度": pivot2.index.astype(str),
            "基期": pivot2["base"].round(2).to_numpy(),
            "本期": pivot2["current"].round(2).to_numpy(),
        })
        drill_table["变化"] = (drill_table["本期"] - drill_table["基期"]).round(2)
        if additive and delta != 0:
            drill_table["贡献%"] = (drill_table["变化"] * direction / abs(delta) * 100).round(1)
        drill_table = drill_table.reindex(
            drill_table["变化"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        drilled_total = float(drill_table["变化"].sum())
        drill = {
            "dim": str(dim2),
            "item": top_item,
            "item_delta": round(item_delta, 2),
            "table": drill_table,
            # 二级分项加总必须回到该项的变化，否则说明下钻时漏了行
            "closure_ok": abs(drilled_total - item_delta) <= 0.02,
        }

    notes = [year_note] if year_note else []
    if missing_base:
        notes.append(f"「{'、'.join(missing_base[:3])}」只在「{intent.get('current')}」有数据，"
                     f"按新增处理（基期记 0）")
    if missing_current:
        notes.append(f"「{'、'.join(missing_current[:3])}」只在「{intent.get('base')}」有数据，"
                     f"按消失处理（本期记 0）")
    if intent.get("kind") == "period" and not intent.get("assumed"):
        days_base = _month_days(intent.get("base_key"))
        days_current = _month_days(intent.get("current_key"))
        # 31 天和 30 天差 3%，算不上偏差，只会添噪；超过 8% 才值得提醒（如 2 月对 3 月）
        if days_base and days_current and abs(days_base - days_current) / max(days_base, days_current) >= 0.08:
            notes.append(f"两期自然天数不同（{days_base} 天 / {days_current} 天），"
                         f"总额未按天数归一，跨期对比要注意天数影响")
    if intent.get("assumed"):
        notes.append(f"问题里只提到一个时段，默认与上一期（{intent.get('base')}）比较")

    # 用户点名的第二层若在结果里找不到，说明为什么没做二级拆解 ——
    # 静默少给一层，用户会以为"已经拆到最细了"
    asked = [h for h in hints if h and str(comp_col) not in h]
    if len(asked) >= 2 and not dim2:
        notes.append(f"问题里点了「{asked[0]}」「{asked[1]}」两层，"
                     f"但结果里只有「{dim}」这一层能拆，已按单层给出")

    result = {
        "ok": True,
        "compare_col": str(comp_col),
        "dim": str(dim),
        "dim2": str(dim2 or ""),
        "metric": str(metric),
        "base_label": str(intent.get("base")),
        "current_label": str(intent.get("current")),
        "base_total": round(base_total, 2),
        "current_total": round(current_total, 2),
        "delta": delta,
        "pct": round(delta / base_total, 4) if base_total else None,
        "table": table,
        "drill": drill,
        "additive": additive,
        "notes": notes,
    }
    result["summary"] = format_attribution(result)
    return result


# ---------------------------------------------------------------- 结论文案
def _num(value) -> str:
    return f"{float(value):,.2f}"


def _pct(value) -> str:
    return f"{value * 100:+.1f}%"


def format_attribution(att: dict) -> str:
    """生成归因结论：数字全部来自 decompose 的计算结果，不经过模型。"""
    if not att or not att.get("ok", True):
        return (att or {}).get("reason") or "无法完成归因。"

    metric, dim = att["metric"], att["dim"]
    base, current = att["base_label"], att["current_label"]
    delta = float(att["delta"])
    table = att["table"]

    if delta > 0:
        verb = "增加"
    elif delta < 0:
        verb = "减少"
    else:
        verb = "持平"

    head = f"「{current}」的{metric}为 {_num(att['current_total'])}"
    if delta == 0:
        head += f"，与「{base}」的 {_num(att['base_total'])} 基本持平。"
    else:
        head += f"，比「{base}」的 {_num(att['base_total'])} {verb} {_num(abs(delta))}"
        if att.get("pct") is not None:
            head += f"（{_pct(att['pct'])}）"
        head += "。"

    up = table[table["变化"] > 0].head(3)
    down = table[table["变化"] < 0].head(3)

    def fmt_rows(rows):
        lines = []
        for _, row in rows.iterrows():
            item = (f"  · {row['维度']}  {row['变化']:+,.2f}"
                    f"（基期 {row['基期']:,.2f} → 本期 {row['本期']:,.2f}）")
            share = row.get("贡献%")
            if att.get("additive") and share is not None and not pd.isna(share):
                if share >= 0:
                    item += f"，占{'增幅' if delta > 0 else '降幅'}的 {share:.1f}%"
                else:
                    item += f"，对冲了 {abs(share):.1f}%"
            lines.append(item)
        return lines

    lines = [head]
    if delta == 0:
        lines.append(f"总额没变，但按「{dim}」看结构有调整：")
    elif up.empty and down.empty:
        lines.append(f"按「{dim}」拆开后各项变化都很小，没有明显的单一来源。")
    else:
        # 先讲"与总方向一致"的那一侧：总额在跌就先把拖累项摆出来，
        # 反过来把上涨项放前面会让人误以为结论是"在涨"。
        if delta < 0:
            blocks = [(down, "主要拖累"), (up, "反向拉动")]
        else:
            blocks = [(up, "主要拉动"), (down, "反向拉低")]
        for rows, label in blocks:
            if rows.empty:
                continue
            lines.append(f"{label}（{len(rows)} 项，合计 {rows['变化'].sum():+,.2f}）：")
            lines.extend(fmt_rows(rows))

    # 二级拆解单独成段：一级结论已经给出"哪一项在动"，这里回答"这一项内部是谁在动"。
    drill = att.get("drill")
    if drill:
        # 二级占比沿用"对总额的贡献"这一口径（与一级同一把尺子），
        # 所以它不会加到 100%，而是加到"该项自己的占比" —— 说清楚免得被当成算错
        share_note = "（各行占比仍是对总额的贡献）" if att.get("additive") else ""
        lines.append("")
        lines.append(f"二级拆解：把变化最大的「{drill['item']}」再按「{drill['dim']}」拆开"
                     f"{share_note}：")
        for _, row in drill["table"].iterrows():
            item = (f"  · {row['维度']}  {row['变化']:+,.2f}"
                    f"（基期 {row['基期']:,.2f} → 本期 {row['本期']:,.2f}）")
            share = row.get("贡献%")
            if att.get("additive") and share is not None and not pd.isna(share):
                if share >= 0:
                    item += f"，占总{'增幅' if delta > 0 else '降幅'}的 {share:.1f}%"
                else:
                    item += f"，对冲了 {abs(share):.1f}%"
            lines.append(item)
        drilled = drill["table"]["变化"].sum()
        if drill.get("closure_ok"):
            lines.append(f"  校验：以上分项加总 = {drilled:+,.2f}，"
                         f"与「{drill['item']}」本身的变化 {drill['item_delta']:+,.2f} 一致。")
        else:
            lines.append(f"  注意：以上分项加总 {drilled:+,.2f} 与「{drill['item']}」本身的"
                         f"变化 {drill['item_delta']:+,.2f} 对不上，二级拆解仅供参考。")

    lines.append("")
    if att.get("additive"):
        lines.append(f"校验：{len(table)} 个{dim}分项的变化加总 = "
                     f"{table['变化'].sum():+,.2f}，与总差额 {delta:+,.2f} 一致，可逐项核对。")
    else:
        lines.append(f"说明：{metric}是比率类指标，{len(table)} 个{dim}的分项变化不能相加，"
                     f"上面按变化幅度排序，只用来看方向。")

    for note in att.get("notes") or []:
        lines.append(f"注：{note}")
    return "\n".join(lines)


def describe_comparison(intent: dict) -> str:
    """把意图写成一句给模型看的说明：谁跟谁比、按什么拆。

    只说"为什么 6 月比 5 月高"时，模型可能只返回两行总数 —— 那样的表拆不开。
    在提问末尾把"要几列交叉表"说清楚，比在 system 里写十遍规则都有效。

    用户点名两个维度时要求**四列**（对比列 × 一级 × 二级 × 指标）：
    多要一列就能在代码侧做二级下钻，而不必再发一次模型调用。
    """
    base, current = intent.get("base"), intent.get("current")
    dims = [d for d in (intent.get("dimension_hints")
                        or [intent.get("dimension_hint", "")]) if d]
    compare = "月份" if intent.get("kind") == "period" else "用户提到的对比维度"
    if len(dims) >= 2:
        shape = (f"「{compare} × {dims[0]} × {dims[1]} × 指标」四列交叉表"
                 f"（{dims[0]} 为一级维度、{dims[1]} 为二级维度）")
    else:
        dim = dims[0] if dims else "最能解释差异的业务分类字段"
        shape = f"「{compare} × {dim} × 指标」三列交叉表"
    return (f"\n\n【归因要求】以「{base}」为参照、「{current}」为被解释对象；"
            f"结果需要{shape}，不要只给两行总数。")


def attribution_chart_frame(att: dict) -> Optional[pd.DataFrame]:
    """归因图专用数据：只留「维度 + 变化」，避免把基期/本期当主指标画错。"""
    if not att or not att.get("ok", True) or att.get("table") is None:
        return None
    table = att["table"]
    if table.empty:
        return None
    return table[["维度", "变化"]].copy()
