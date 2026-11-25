"""相对时间锚点（P0）离线用例：不调用 LLM。

覆盖三类：
1. 纯函数解析：各类相对时间 → 显式区间，今天可注入、覆盖范围判定；
2. 两条引擎的拒答与注入：无交集直接说明（不花钱），有交集进提示词；
3. 反例：裸数字、明确年月、LIMIT 6 不得被当成相对时间。
"""
import unittest
from datetime import date
from unittest import mock

from app.agent import DataAgent
from app.context import AgentContext
from app.nodes.understand import make_understand_node
from app.state import AgentState
from app.tools.time_anchor import (
    format_time_hint, out_of_range_answer, resolve_time_anchor, to_date,
)

TODAY = "2026-11-25"
COVER = {"min": "2025-01-01", "max": "2025-12-31"}
IN_COVER = "2025-07-15"


def first(question, today=TODAY, coverage=COVER):
    found = resolve_time_anchor(question, today=today, coverage=coverage)
    return found[0] if found else None


class TestResolve(unittest.TestCase):
    def test_last_month_full_calendar_month(self):
        a = first("上个月的销售额是多少？")
        self.assertEqual(a.span, (date(2026, 10, 1), date(2026, 10, 31)))
        self.assertEqual(a.phrase, "上个月")
        self.assertTrue(a.out_of_range, "库只到 2025-12，2026-10 完全没交集")

    def test_last_month_in_coverage(self):
        a = first("上个月销售额", today=IN_COVER)
        self.assertEqual(a.span, (date(2025, 6, 1), date(2025, 6, 30)))
        self.assertFalse(a.out_of_range)
        self.assertFalse(a.partial)

    def test_recent_days_inclusive(self):
        """最近 N 天含今天，30 天应是 [today-29, today] 而不是 MAX(date)-29。"""
        a = first("最近30天的销售额", today=IN_COVER)
        self.assertEqual(a.span, (date(2025, 6, 16), date(2025, 7, 15)))
        self.assertEqual(a.phrase, "最近30天")

    def test_recent_days_chinese_number(self):
        a = first("最近七天订单")
        self.assertEqual(a.span, (date(2026, 11, 19), date(2026, 11, 25)))

    def test_recent_months(self):
        a = first("最近3个月的走势", today=IN_COVER)
        self.assertEqual(a.span, (date(2025, 5, 1), date(2025, 7, 31)))

    def test_recent_half_year(self):
        a = first("最近半年", today=IN_COVER)
        self.assertEqual(a.span, (date(2025, 2, 1), date(2025, 7, 31)))

    def test_quarters(self):
        last, this = resolve_time_anchor("上季度比这个季度", today=IN_COVER, coverage=COVER)
        self.assertEqual(last.span, (date(2025, 4, 1), date(2025, 6, 30)))
        self.assertEqual(this.span, (date(2025, 7, 1), date(2025, 9, 30)))

    def test_half_year(self):
        """半年必须换算出自然半年，且不能被「今年/去年」抢先按全年算。

        「今年上半年」曾经命中的是 this_year，系统会注入「今年 = 全年」——
        比不认半年更糟：模型会照着答全年，还理直气壮。
        """
        cases = [
            ("今年上半年销售额", (date(2026, 1, 1), date(2026, 6, 30))),
            ("今年下半年销售额", (date(2026, 7, 1), date(2026, 12, 31))),
            ("去年上半年", (date(2025, 1, 1), date(2025, 6, 30))),
            ("去年 下半年", (date(2025, 7, 1), date(2025, 12, 31))),
            ("前年下半年订单", (date(2024, 7, 1), date(2024, 12, 31))),
            ("去年 H2 的订单", (date(2025, 7, 1), date(2025, 12, 31))),
        ]
        for q, expected in cases:
            with self.subTest(question=q):
                found = resolve_time_anchor(q, today=TODAY, coverage={})
                self.assertEqual(len(found), 1, found)
                self.assertEqual(found[0].span, expected)

    def test_half_year_wins_over_year_word(self):
        """「去年上半年」不能再拆出一个「去年」，否则两个锚点互相矛盾。"""
        found = resolve_time_anchor("去年上半年比今年上半年", today=TODAY, coverage={})
        self.assertEqual([a.phrase for a in found], ["去年上半年", "今年上半年"])

    def test_bare_half_year_has_no_anchor(self):
        """裸「上半年」不猜年份：它本来就是字面条件，归 scope_guard 管。"""
        for q in ["上半年的销售额", "2025 年上半年的销售额", "下半年订单"]:
            with self.subTest(question=q):
                self.assertEqual(resolve_time_anchor(q, today=TODAY, coverage=COVER), [])

    def test_year_and_this_month(self):
        cases = [
            ("去年销售额", (date(2025, 1, 1), date(2025, 12, 31))),
            ("今年销售额", (date(2026, 1, 1), date(2026, 12, 31))),
            ("本月订单", (date(2026, 11, 1), date(2026, 11, 30))),
            ("前天订单", (date(2026, 11, 23), date(2026, 11, 23))),
        ]
        for q, expected in cases:
            with self.subTest(question=q):
                self.assertEqual(first(q).span, expected)

    def test_long_phrase_wins_over_short(self):
        """「去年同月」不可再拆出一个「去年」——两个锚点会互相矛盾。"""
        found = resolve_time_anchor("去年同月比今年同月", today=IN_COVER, coverage=COVER)
        phrases = [a.phrase for a in found]
        self.assertIn("去年同月", phrases)
        self.assertNotIn("去年", phrases, "短词应与长词位置重叠被丢弃")

    def test_explicit_date_has_no_anchor(self):
        """明确年月不产生锚点：它本来就是字面条件，归 scope_guard 管。"""
        for q in ["2025年6月销售额", "2025-Q2 的订单", "3-5 月的销量"]:
            with self.subTest(question=q):
                self.assertEqual(resolve_time_anchor(q, today=TODAY, coverage=COVER), [])

    def test_bare_numbers_not_relative(self):
        """裸数字、LIMIT 6、行号等不能误判成相对时间（回归坑）。"""
        for q in ["一共多少订单 LIMIT 6", "销量前 10 名", "第 3 季度销售额"]:
            with self.subTest(question=q):
                self.assertEqual(resolve_time_anchor(q, today=TODAY, coverage=COVER), [])

    def test_no_anchor_for_empty(self):
        self.assertEqual(resolve_time_anchor("", today=TODAY, coverage=COVER), [])

    def test_to_date_accepts_date_and_string(self):
        self.assertEqual(to_date(date(2025, 6, 1)), date(2025, 6, 1))
        self.assertEqual(to_date("2025-06-01"), date(2025, 6, 1))
        self.assertEqual(to_date("not-a-date"), date.today())

    def test_partial_overlap(self):
        """区间一部分在库内：标记 partial，不是 out_of_range（不误拒正常问题）。"""
        a = first("上个月销售额", today="2025-02-10",
                  coverage={"min": "2025-01-15", "max": "2025-12-31"})
        self.assertEqual(a.span, (date(2025, 1, 1), date(2025, 1, 31)))
        self.assertFalse(a.out_of_range)
        self.assertTrue(a.partial)
        self.assertIn("2025-01-01", a.gap)

    def test_no_coverage_means_no_verdict(self):
        """拿不到覆盖范围时不判越界（宁可漏判，不可误拒）。"""
        a = first("上个月销售额", today=TODAY, coverage={})
        self.assertFalse(a.out_of_range)


class TestHintAndRefusal(unittest.TestCase):
    def test_hint_contains_today_and_span(self):
        found = resolve_time_anchor("上个月的销售额", today=IN_COVER, coverage=COVER)
        hint = format_time_hint(found, today=IN_COVER, coverage=COVER)
        self.assertIn("今天是 2025-07-15", hint)
        self.assertIn("2025-06-01 ~ 2025-06-30", hint)
        self.assertIn("库里数据覆盖", hint)

    def test_no_hint_without_anchor(self):
        self.assertEqual(format_time_hint([], today=TODAY, coverage=COVER), "")

    def test_refusal_has_no_alternative_range(self):
        found = resolve_time_anchor("上个月的销售额", today=TODAY, coverage=COVER)
        answer = out_of_range_answer(found, COVER)
        self.assertIn("没有数据", answer)
        self.assertIn("没有把时间平移", answer)
        self.assertNotIn("2025-06", answer, "不能给出替代月份，哪怕是暗示")

    def test_no_refusal_when_in_range(self):
        found = resolve_time_anchor("上个月的销售额", today=IN_COVER, coverage=COVER)
        self.assertEqual(out_of_range_answer(found, COVER), "")


class TestIntegration(unittest.TestCase):
    def test_understand_marks_time_defer(self):
        ctx = AgentContext()
        with mock.patch.object(type(ctx), "today", property(lambda self: date(2026, 11, 25))):
            node = make_understand_node(ctx)
            state = node(AgentState(question="上个月的销售额是多少？", history=[]))
        self.assertIn("库里没有数据", state["time_defer"])
        self.assertIn("今天是 2026-11-25", state["terms"])

    def test_understand_injects_anchor_into_terms(self):
        ctx = AgentContext()
        with mock.patch.object(type(ctx), "today", property(lambda self: date(2025, 7, 15))):
            state = make_understand_node(ctx)(AgentState(
                question="最近30天销售额", history=[]))
        self.assertFalse(state["time_defer"])
        self.assertIn("2025-06-16 ~ 2025-07-15", state["terms"])
        self.assertIn("今天是 2025-07-15", state["terms"])

    def test_graph_refuses_without_llm(self):
        """图引擎：无交集走 clarify，不调生成（用 mock 保证没碰模型）。"""
        from app.graph import build_graph, route_after_understand

        with mock.patch.object(type(AgentContext()), "today",
                               property(lambda self: date(2026, 11, 25))):
            state = make_understand_node(AgentContext())(
                AgentState(question="上个月的销售额是多少？", history=[]))
        self.assertEqual(route_after_understand(state), "clarify")
        with mock.patch("app.graph.generate_sql", side_effect=AssertionError("不应调 LLM")):
            app = build_graph()
            out = app.invoke(AgentState(question="上个月的销售额是多少？", history=[],
                                        trace_id="test-time-anchor"))
        self.assertIn("没有数据", out["answer"])
        self.assertIsNone(out["result"])
        self.assertEqual(out["chart_type"], "none")

    def test_linear_refuses_without_llm(self):
        agent = DataAgent()
        with mock.patch.object(type(agent.ctx), "today",
                               property(lambda self: date(2026, 11, 25))), \
             mock.patch("app.agent.generate_sql",
                        side_effect=AssertionError("不应调 LLM")):
            res = agent.ask("上个月的销售额是多少？")
        self.assertFalse(res.ok)
        self.assertIn("没有数据", res.refuse_reason)
        self.assertEqual(res.statement, "")


if __name__ == "__main__":
    unittest.main()
