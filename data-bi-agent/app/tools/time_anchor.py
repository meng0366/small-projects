"""相对时间锚点（P0，2026-11-25）。

为什么需要这一层
----------------
实测两个问题：

* 「上个月的销售额」→ 模型自己编了个"今天 = 2025-12-31"，写
  `strftime('%Y-%m', '2025-12-31', '-1 month')`，答的是 2025-11；
* 「最近 30 天」→ 模型把口径悄悄换成"数据里的最后 30 天"，写
  `>= date((SELECT MAX(order_date) FROM orders), '-29 days')`。

两种错法不同、根因相同：**系统从没告诉模型今天是哪天**。用户话里的相对时间
隐含一个参考点，模型只能自己猜；而这类问题没有任何年份/月份字面，
`scope_guard` 的锚点集是空的，完全看不见。

设计原则
--------
1. **今天由系统给**，不是模型猜：`resolve_time_anchor(..., today=...)` 可注入，
   默认取当天；测试用固定日期，避免"今天"变化让用例漂移。
2. **代码换算成显式区间**，不是提示词请求模型换算：相对时间 → `[start, end]`
   全用 stdlib `calendar` / `datetime` 算，零 token、可离线穷举。
3. **无交集就是没有数据**，不许平移到有数据的区间（那是"换个问题再答"）。
   部分超出才按覆盖内的部分回答，并明说超出情况。
4. 只做**公历自然周期**：上个月 = 上一自然月整月，最近 N 天含今天，
   季度按自然季度，半年按自然半年（上半年 1-6 月）。中文"去年""前天"等都按同一规则。

本模块是纯函数，不依赖 LLM / 数据库 / 配置。
"""
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Optional, Sequence, Tuple


_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_int(text: str) -> Optional[int]:
    """把「三」「十二」这类中文数字转成 int；纯阿拉伯数字也认。"""
    if text is None:
        return None
    if text.isdigit():
        return int(text)
    if text in _CN_NUM:
        return _CN_NUM[text]
    if text.startswith("十"):
        tail = text[1:]
        return 10 + (_CN_NUM.get(tail, 0) if tail else 0)
    if "十" in text:
        head, tail = text.split("十", 1)
        return _CN_NUM.get(head, 1) * 10 + _CN_NUM.get(tail, 0)
    return None


@dataclass(frozen=True)
class TimeAnchor:
    """一处相对时间的解析结果。日期都是含端点的闭区间。"""
    phrase: str             # 用户原文里命中的说法
    start: date             # 起始日（含）
    end: date               # 结束日（含）
    label: str              # 人话描述，用于提示词与回答
    out_of_range: bool      # 与数据覆盖完全无交集
    partial: bool           # 有交集，但有一部分超出覆盖
    gap: str                # 超出部分的说明（无超出时为空字符串）

    @property
    def span(self) -> Tuple[date, date]:
        return self.start, self.end


def to_date(value) -> date:
    """把 `date` / 'YYYY-MM-DD' 统一成 date；无效值退回今天。"""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return date.today()


def _month_bounds(year: int, month: int) -> Tuple[date, date]:
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _shift_month(day: date, delta: int) -> Tuple[int, int]:
    """月份加减，返回 (年, 月)。"""
    total = day.year * 12 + day.month - 1 + delta
    return total // 12, total % 12 + 1


def _quarter_bounds(year: int, quarter: int) -> Tuple[date, date]:
    start_month = (quarter - 1) * 3 + 1
    return _month_bounds(year, start_month)[0], _month_bounds(year, start_month + 2)[1]


def _half_bounds(year: int, half: int) -> Tuple[date, date]:
    """上半年 = 1-6 月，下半年 = 7-12 月，都按自然半年。"""
    if half == 1:
        return date(year, 1, 1), date(year, 6, 30)
    return date(year, 7, 1), date(year, 12, 31)


def _week_bounds(day: date) -> Tuple[date, date]:
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def _coverage(coverage) -> Optional[Tuple[date, date]]:
    """coverage 支持 'min'/'max' dict、(start, end) 元组或 None。"""
    if not coverage:
        return None
    if isinstance(coverage, dict):
        lo, hi = coverage.get("min"), coverage.get("max")
    else:
        lo, hi = (list(coverage) + [None, None])[:2]
    if not lo or not hi:
        return None
    try:
        return to_date(lo), to_date(hi)
    except (TypeError, ValueError):
        return None


def _fit(phrase: str, start: date, end: date, coverage) -> TimeAnchor:
    """比对覆盖范围，生成锚点。日期是闭区间，日期区间写法用 `< 下界` 也等价。"""
    cov = _coverage(coverage)
    out, partial, gap = False, False, ""
    if cov:
        lo, hi = cov
        if end < lo or start > hi:
            out = True
            gap = (f"用户要求的 {start.isoformat()} ~ {end.isoformat()} "
                   f"与库里覆盖的 {lo.isoformat()} ~ {hi.isoformat()} 完全没有交集")
        elif start < lo or end > hi:
            partial = True
            sides = []
            if start < lo:
                sides.append(f"{start.isoformat()} ~ {(lo - timedelta(days=1)).isoformat()}")
            if end > hi:
                sides.append(f"{(hi + timedelta(days=1)).isoformat()} ~ {end.isoformat()}")
            gap = "、".join(sides)
    label = f"{start.isoformat()} ~ {end.isoformat()}"
    return TimeAnchor(phrase, start, end, label, out, partial, gap)


# 各类说法的正则。顺序有讲究：更具体的先匹配（"去年同月"先于"去年"）。
_NUM = r"([0-9]+|[零一二两三四五六七八九十]+)"
_RECENT = r"(?:最近|近|过去|最近的一?)\s*"
_PATTERNS = [
    # 长词在前：命中「去年同月」后，同一位置的「去年」会被去重逻辑丢掉。
    ("last_year_same", re.compile(r"(去年同月|上年同月|去年同期|上年同期)")),
    ("last_month", re.compile(r"(上个月|上月|上一个自然月|前一个月)")),
    ("next_month", re.compile(r"(下个月|下月|下一个月)")),
    ("this_month", re.compile(r"(这个月|本月|当月|本月份)")),
    ("last_quarter", re.compile(r"(上个季度|上季度|上一个季度|前一?季度)")),
    ("this_quarter", re.compile(r"(这个季度|本季度|当季)")),
    # 半年必须排在「去年/今年」之前：否则「今年上半年」里的「今年」先命中，
    # 系统会把半年的问题按全年注入锚点 —— 比不认半年更糟（模型会照着答全年）。
    # 只认带年份词的说法；裸「上半年」归 scope_guard 的字面锚点管，不在这里猜年份。
    ("half_year", re.compile(
        r"(?P<yp>今年|本年度|本年|上一年|上年|去年|前年)\s*(?:的)?\s*"
        r"(?P<hp>上半年|下半年|前半年|后半年|H(?P<hn>[12])(?!\d))", re.IGNORECASE)),
    ("last_year", re.compile(r"(去年|上一年|前一年)")),
    ("this_year", re.compile(r"(今年|本年|本年度|当年)")),
    ("last_week", re.compile(r"(上周|上个星期|前一周)")),
    ("this_week", re.compile(r"(本周|这周|这个星期)")),
    ("before_yesterday", re.compile(r"(前天)")),
    ("yesterday", re.compile(r"(昨天|昨日)")),
    ("today", re.compile(r"(今天|当日)")),
    # 带前缀才认：裸「6 个月」「LIMIT 6」不是相对时间，误判会拦掉正常问题。
    ("recent_months", re.compile(_RECENT + _NUM + r"\s*个?月(?:以内|之内|内)?")),
    ("recent_days", re.compile(_RECENT + _NUM + r"\s*(?:天|日)(?:以内|之内|内)?")),
    ("recent_half_year", re.compile(_RECENT + r"半\s*年")),
]


def has_relative_time(question: str) -> bool:
    """问题里是否出现相对时间说法（不换算，只看有没有）。

    给 SQL 复用判断"要不要把今天绑进缓存键"用：含相对时间的问题，SQL 里已经
    被换算成显式日期区间，换一天再复用就是拿上个月的数冒充这个月。

    刻意复用 `_PATTERNS` 同一份规则表，而不是另写一个关键词列表 ——
    两处判据一旦分家就会漂移（"近两周"加进锚点、却忘了加进缓存键，
    于是跨天复用一个错的 SQL，这种错还没有任何报错）。
    """
    text = (question or "").strip()
    if not text:
        return False
    return any(regex.search(text) for _, regex in _PATTERNS)


def resolve_time_anchor(question: str, today=None, coverage=None) -> list:
    """抽出问题里的相对时间，换算成显式日期区间。

    返回按出现顺序排列的锚点列表；没有任何相对时间时返回空列表
    （绝大多数问题都走这里，零额外开销）。同一说法重复出现只记一次。
    """
    text = (question or "").strip()
    if not text:
        return []
    base = to_date(today)
    found: list = []
    # 记录已被长词占用的字符位置：[start, end)，短词与之重叠就跳过。
    # 只按 phrase 去重不够：「去年同月」与「去年」文本不同，却指同一段时间的不同粒度，
    # 一起注入会让模型看到两个互相矛盾的说法。
    occupied: list = []

    # 「去年同月」这类长词必须先于「去年」，所以按 pattern 顺序逐个扫。
    # 用 finditer 而不是 search，便于一个句子里出现两个时点（如"上月比上上月"）。
    for kind, regex in _PATTERNS:
        for match in regex.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            phrase = match.group(0).strip()
            if kind == "last_month":
                y, m = _shift_month(base, -1)
                s, e = _month_bounds(y, m)
            elif kind == "this_month":
                s, e = _month_bounds(base.year, base.month)
            elif kind == "next_month":
                y, m = _shift_month(base, 1)
                s, e = _month_bounds(y, m)
            elif kind == "last_quarter":
                y, m = _shift_month(base, -3)
                s, e = _quarter_bounds(y, (m - 1) // 3 + 1)
            elif kind == "this_quarter":
                s, e = _quarter_bounds(base.year, (base.month - 1) // 3 + 1)
            elif kind == "last_year_same":
                y, m = _shift_month(base, -12)
                s, e = _month_bounds(y, m)
            elif kind == "half_year":
                year_phrase = match.group("yp")
                if year_phrase == "前年":
                    year = base.year - 2
                elif year_phrase in ("今年", "本年", "本年度"):
                    year = base.year
                else:
                    year = base.year - 1
                if match.group("hn"):
                    half = int(match.group("hn"))
                else:
                    half = 1 if match.group("hp") in ("上半年", "前半年") else 2
                s, e = _half_bounds(year, half)
            elif kind == "last_year":
                s, e = date(base.year - 1, 1, 1), date(base.year - 1, 12, 31)
            elif kind == "this_year":
                s, e = date(base.year, 1, 1), date(base.year, 12, 31)
            elif kind == "last_week":
                s, e = _week_bounds(base - timedelta(days=7))
            elif kind == "this_week":
                s, e = _week_bounds(base)
            elif kind == "yesterday":
                s = e = base - timedelta(days=1)
            elif kind == "today":
                s = e = base
            elif kind == "before_yesterday":
                s = e = base - timedelta(days=2)
            elif kind == "recent_days":
                n = _cn_int(match.group(1))
                if not n or n < 1 or n > 3650:
                    continue
                s, e = base - timedelta(days=n - 1), base
            elif kind == "recent_months":
                n = _cn_int(match.group(1))
                if not n or n < 1 or n > 120:
                    continue
                y, m = _shift_month(base, -(n - 1))
                s = _month_bounds(y, m)[0]
                e = _month_bounds(base.year, base.month)[1]
            elif kind == "recent_half_year":
                y, m = _shift_month(base, -5)
                s = _month_bounds(y, m)[0]
                e = _month_bounds(base.year, base.month)[1]
            else:
                continue
            found.append(_fit(phrase, s, e, coverage))
    return found


def format_time_hint(anchors: Sequence[TimeAnchor], today=None,
                     coverage=None) -> str:
    """拼成注入提示词的片段。没有锚点时返回空字符串（不占 token）。"""
    if not anchors:
        return ""
    base = to_date(today)
    cov = _coverage(coverage)
    lines = [
        "【时间锚点：系统换算，按此执行，不要自行推断日期】",
        f"今天是 {base.isoformat()}（由系统提供，不是从数据里推出来的）。",
    ]
    for item in anchors:
        lines.append(f"用户说的「{item.phrase}」= {item.label}。")
        if item.gap:
            lines.append(f"注意：{item.gap}。")
    if cov:
        lines.append(f"库里数据覆盖 {cov[0].isoformat()} ~ {cov[1].isoformat()}。")
    return "\n".join(lines) + "\n"


def out_of_range_answer(anchors: Sequence[TimeAnchor], coverage=None) -> str:
    """完全无交集时的说明：直接说没有数据，不给任何替代区间。"""
    bad = [a for a in anchors if a.out_of_range]
    if not bad:
        return ""
    cov = _coverage(coverage)
    lines = ["这个问题问的时间段，库里没有数据。"]
    for item in bad:
        lines.append(f"你问的「{item.phrase}」是 {item.label}，"
                     f"这段时间不在库的覆盖范围内。")
    if cov:
        lines.append(f"库里实际覆盖 {cov[0].isoformat()} ~ {cov[1].isoformat()}。")
    lines.append("没有把时间平移到别的区间来凑数据 —— 那样算出来的数字不是这个问题的答案。"
                 "请把问题改成明确的年月再问。")
    return "\n".join(lines)
