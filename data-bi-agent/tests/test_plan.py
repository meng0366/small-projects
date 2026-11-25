"""计划先行（P1）的离线测试：全部不调用 LLM。

覆盖四层：
1. `needs_plan`：哪些问题值得多花一次调用（简单问题必须跳过，否则白烧 token）；
2. `parse_plan`：模型输出的各种脏格式能不能取出来；
3. `validate_plan`：三类硬拦截（时间无交集 / 取值不存在 / 年份被换）与两类提醒；
4. 端到端：致命问题真的短路到澄清、正常计划真的进了生成提示、解析失败不影响回答。
"""
import json
import unittest
from unittest import mock

import pandas as pd

import app.agent as A
import app.graph as G
import app.nodes.plan as P
from app.tools.plan_guard import (
    format_plan_hint,
    format_plan_summary,
    has_fatal,
    needs_plan,
    parse_plan,
    validate_plan,
)

TABLES = {
    "orders": ["order_id", "order_date", "customer_id", "product_id", "quantity",
               "unit_price", "discount", "amount"],
    "customers": ["customer_id", "customer_name", "region", "customer_level"],
    "products": ["product_id", "product_name", "category", "unit_price"],
}
ENUMS = [
    {"table": "customers", "column": "region", "label": "所在区域",
     "values": ["华东", "华南", "华北", "西南", "华中"], "hints": ["地区", "区域"]},
    {"table": "products", "column": "category", "label": "商品品类",
     "values": ["手机", "笔记本", "平板", "配件", "智能家居"], "hints": ["品类", "类"]},
]
COVERAGE = {"min": "2025-01-01", "max": "2025-12-31"}
GOOD_SQL = ("SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额 "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.region")


def plan_of(**overrides) -> dict:
    base = {
        "intent": "求各区域销售额",
        "tables": ["orders", "customers"],
        "metric": {"name": "销售额", "agg": "SUM", "column": "amount"},
        "dimensions": ["region"],
        "filters": [],
        "time_range": {"column": "order_date", "start": "2025-06-01", "end": "2025-06-30"},
        "assumptions": [],
        "needs_clarify": "",
    }
    base.update(overrides)
    return base


class TestNeedsPlan(unittest.TestCase):
    def test_simple_question_skips(self):
        """简单问题必须跳过：多一次调用是实打实的钱。"""
        ok, reason = needs_plan("各区域的销售额是多少？")
        self.assertFalse(ok)
        self.assertIn("简单", reason)

    def test_attribution_triggers(self):
        ok, reason = needs_plan("为什么 6 月的销售额比 5 月高？",
                                intent={"base": "5月", "current": "6月"})
        self.assertTrue(ok)
        self.assertIn("归因", reason)

    def test_history_triggers(self):
        """多轮追问里的省略说法最容易改错对象。"""
        ok, _ = needs_plan("只看华东的", has_history=True)
        self.assertTrue(ok)

    def test_relative_time_triggers(self):
        ok, reason = needs_plan("上个月的销售额是多少？", anchors=[object()])
        self.assertTrue(ok)
        self.assertIn("相对时间", reason)

    def test_filter_word_triggers(self):
        ok, reason = needs_plan("只看华东的销售额")
        self.assertTrue(ok)
        self.assertIn("过滤", reason)

    def test_ratio_word_triggers(self):
        ok, reason = needs_plan("各品类的销售额占比是多少")
        self.assertTrue(ok)
        self.assertIn("比率", reason)

    def test_long_question_triggers(self):
        question = "帮我看看华东区域里重点客户买的手机这个品类在最近几个季度的销售额走势"
        ok, reason = needs_plan(question)
        self.assertTrue(ok)
        self.assertIn("较长", reason)

    def test_pandas_route_skips(self):
        """单表路线不启用：列的上下文太少，计划没法校验，白花一次调用。"""
        ok, reason = needs_plan("只看华东的销售额", route="pandas")
        self.assertFalse(ok)
        self.assertIn("单表", reason)


class TestParsePlan(unittest.TestCase):
    def test_plain_json(self):
        plan = parse_plan(json.dumps(plan_of(), ensure_ascii=False))
        self.assertEqual(plan["metric"]["column"], "amount")
        self.assertEqual(plan["dimensions"], ["region"])

    def test_code_fence_and_chatter(self):
        """模型爱加围栏、爱先说一句 —— 取第一个 { 到最后一个 } 就行。"""
        raw = "好的，计划如下：\n```json\n" + json.dumps(plan_of(), ensure_ascii=False) + "\n```"
        self.assertIsNotNone(parse_plan(raw))

    def test_missing_fields_are_filled(self):
        """字段缺失要补空而不是整份丢掉 —— 剩下的字段仍能用来校验。"""
        plan = parse_plan('{"metric": {"column": "amount"}}')
        self.assertEqual(plan["dimensions"], [])
        self.assertEqual(plan["filters"], [])
        self.assertEqual(plan["time_range"]["start"], "")

    def test_filters_normalized(self):
        raw = json.dumps({"filters": [{"column": "region", "values": "华东"},
                                      {"nope": 1}]}, ensure_ascii=False)
        plan = parse_plan(raw)
        self.assertEqual(len(plan["filters"]), 1)
        self.assertEqual(plan["filters"][0]["values"], ["华东"])

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_plan("我不知道怎么计划"))
        self.assertIsNone(parse_plan("{不是 JSON}"))
        self.assertIsNone(parse_plan("[1, 2, 3]"))
        self.assertIsNone(parse_plan(""))


class TestValidatePlan(unittest.TestCase):
    def _check(self, plan, question=""):
        return validate_plan(plan, TABLES, ENUMS, COVERAGE, question)

    def test_clean_plan_has_no_issue(self):
        self.assertEqual(self._check(plan_of()), [])

    def test_unknown_table_is_only_a_warning(self):
        """表名带别名/大小写的情况太多，只提醒不阻断。"""
        issues = self._check(plan_of(tables=["order"]))
        self.assertTrue(issues)
        self.assertTrue(all(item["level"] == "warn" for item in issues))
        self.assertFalse(has_fatal(issues))

    def test_unknown_column_is_only_a_warning(self):
        issues = self._check(plan_of(dimensions=["sales_amount"]))
        self.assertIn("sales_amount", " ".join(i["message"] for i in issues))
        self.assertFalse(has_fatal(issues))

    def test_time_outside_coverage_is_fatal(self):
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2024-03-01", "end": "2024-03-31"})
        issues = self._check(plan)
        self.assertTrue(has_fatal(issues))
        fatal = [i for i in issues if i["level"] == "fatal"][0]
        self.assertEqual(fatal["kind"], "time")
        self.assertIn("没有交集", fatal["message"])
        # 必须写明"没有平移"，这是 P0-2b 那条教训的延续
        self.assertIn("没有把时间平移", fatal["message"])

    def test_time_partial_overlap_is_ok(self):
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-12-01", "end": "2026-01-31"})
        self.assertFalse(has_fatal(self._check(plan)))

    def test_reversed_time_is_warning(self):
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-06-30", "end": "2025-06-01"})
        issues = self._check(plan)
        self.assertFalse(has_fatal(issues))
        self.assertIn("倒置", " ".join(i["message"] for i in issues))

    def test_month_only_dates(self):
        """模型写 '2025-06' 而不是完整日期时，按月初/月末补全。"""
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-06", "end": "2025-06"})
        self.assertFalse(has_fatal(self._check(plan)))

    def test_unknown_enum_value_is_fatal(self):
        plan = plan_of(filters=[{"column": "category", "values": ["办公用品"], "op": "in"}])
        issues = self._check(plan)
        fatal = [i for i in issues if i["level"] == "fatal"]
        self.assertEqual(len(fatal), 1)
        self.assertEqual(fatal[0]["kind"], "enum")
        # payload 要能直接喂给澄清节点
        payload = fatal[0]["payload"]
        self.assertEqual(payload["candidate"], "办公用品")
        self.assertIn("手机", payload["values"])

    def test_known_enum_value_passes(self):
        plan = plan_of(filters=[{"column": "category", "values": ["手机"], "op": "in"}])
        self.assertFalse(has_fatal(self._check(plan)))

    def test_user_year_mismatch_is_fatal(self):
        """用户说 2024，计划却按 2025 算 —— 这是"换了一年数据"，必须拦。"""
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-03-01", "end": "2025-03-31"})
        issues = self._check(plan, question="2024 年 3 月的销售额是多少？")
        fatal = [i for i in issues if i["level"] == "fatal"]
        self.assertTrue(fatal)
        self.assertEqual(fatal[0]["kind"], "anchor")
        self.assertIn("2024", fatal[0]["message"])

    def test_user_year_match_passes(self):
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-03-01", "end": "2025-03-31"})
        self.assertFalse(has_fatal(self._check(plan, question="2025 年 3 月的销售额是多少？")))

    def test_user_month_missing_from_range_is_fatal(self):
        """用户说 6 月，计划却算的是 5 月。"""
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-05-01", "end": "2025-05-31"})
        issues = self._check(plan, question="6 月的销售额是多少？")
        fatal = [i for i in issues if i["level"] == "fatal"]
        self.assertTrue(fatal)
        self.assertIn("6", fatal[0]["message"])

    def test_half_year_widened_plan_is_fatal(self):
        """用户问上半年，计划按全年算：数据一有就悄悄答完，重试守卫根本没机会出场。

        这是半年在 scope_guard 之外补的第二道 —— 计划阶段零 token 就能拦住。
        """
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-01-01", "end": "2025-12-31"})
        issues = self._check(plan, question="2025 年上半年的销售额是多少？")
        fatal = [i for i in issues if i["level"] == "fatal"]
        self.assertTrue(fatal)
        self.assertEqual(fatal[0]["kind"], "anchor")
        self.assertIn("上半年", fatal[0]["message"])
        self.assertIn("没有拿更宽的时段来代替", fatal[0]["message"])

    def test_second_half_starting_in_january_is_fatal(self):
        """下半年从 1 月起 = 范围被放大。方向不同，判据也不同。"""
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-01-01", "end": "2025-12-31"})
        issues = self._check(plan, question="2025 年下半年的销售额是多少？")
        self.assertTrue(any(i["kind"] == "anchor" and "下半年" in i["message"]
                            for i in issues if i["level"] == "fatal"))

    def test_half_year_exact_plan_passes(self):
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-01-01", "end": "2025-06-30"})
        self.assertFalse(has_fatal(self._check(plan, question="2025 年上半年的销售额是多少？")))

    def test_cross_year_plan_is_not_blocked_for_half(self):
        """跨年区间不判半年：边界写法太多，宁可不拦（漏判优于误拦）。"""
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-07-01", "end": "2026-06-30"})
        issues = self._check(plan, question="2025 年下半年的销售额是多少？")
        self.assertFalse(any(i["kind"] == "anchor" for i in issues))

    def test_attribution_not_blocked_by_single_period(self):
        """归因比的是两期，计划只有一个区间 —— 不按单期校验，否则天天误拦。"""
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2025-06-01", "end": "2025-06-30"})
        issues = validate_plan(plan, TABLES, ENUMS, COVERAGE,
                               "6 月的销售额比 5 月高吗？", skip_anchor_check=True)
        self.assertFalse(has_fatal(issues))

    def test_assumption_is_surfaced(self):
        plan = plan_of(assumptions=["用户说「6月」，按 2025 年 6 月计算"])
        issues = self._check(plan)
        self.assertTrue(any(i["kind"] == "assumption" for i in issues))
        self.assertFalse(has_fatal(issues))


class TestFormat(unittest.TestCase):
    def test_hint_contains_key_sections(self):
        text = format_plan_hint(plan_of(assumptions=["按 2025 年 6 月计算"]))
        self.assertIn("查询计划", text)
        self.assertIn("SUM(amount)", text)
        self.assertIn("region", text)
        self.assertIn("2025-06-01", text)
        self.assertIn("按 2025 年 6 月计算", text)
        self.assertIn("不要自行更改", text)

    def test_hint_carries_warnings(self):
        """表名列名对不上的提醒要进提示词，生成时才有机会改。"""
        issues = validate_plan(plan_of(tables=["order"]), TABLES, ENUMS, COVERAGE)
        text = format_plan_hint(plan_of(tables=["order"]), issues)
        self.assertIn("不在库里", text)

    def test_summary_is_one_line(self):
        summary = format_plan_summary(plan_of())
        self.assertIn("SUM(amount)", summary)
        self.assertIn("region", summary)
        self.assertNotIn("\n", summary)

    def test_empty_plan_formats_empty(self):
        self.assertEqual(format_plan_hint(None), "")
        self.assertEqual(format_plan_summary(None), "")


class PlanEndToEnd(unittest.TestCase):
    """端到端：计划真的影响了生成，且只在必要时才多花一次调用。"""

    def test_fatal_plan_blocks_generation(self):
        """取值不存在 → 澄清，不该生成 SQL（否则模型会挑个最近的顶替）。"""
        plan = plan_of(filters=[{"column": "category", "values": ["办公用品"], "op": "in"}])
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        with mock.patch.object(P, "generate_plan",
                               return_value=json.dumps(plan, ensure_ascii=False)), \
                mock.patch.object(G, "generate_sql") as sql:
            state = agent.run("只看办公用品的销售额")

        sql.assert_not_called()
        self.assertIsNone(state["result"])
        self.assertIn("办公用品", state["answer"])
        self.assertIn("手机", state["answer"])       # 列出真实可选值

    def test_fatal_plan_reports_missing_time(self):
        plan = plan_of(time_range={"column": "order_date",
                                   "start": "2024-03-01", "end": "2024-03-31"})
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        with mock.patch.object(P, "generate_plan",
                               return_value=json.dumps(plan, ensure_ascii=False)), \
                mock.patch.object(G, "generate_sql") as sql:
            state = agent.run("只看华东 2024 年 3 月的销售额")

        sql.assert_not_called()
        self.assertIn("没有交集", state["answer"])
        self.assertIn("2025-01-01", state["answer"])

    def test_plan_hint_reaches_generation(self):
        """计划通过校验后，必须作为提示的一部分进到生成调用里。"""
        captured = {}

        def fake_sql(schema, terms, question, previous_error=None, previous_empty=False,
                     lessons=None, plan_hint=""):
            captured["plan_hint"] = plan_hint
            return GOOD_SQL

        agent = G.GraphAgent(with_chart=False, with_insight=False)
        with mock.patch.object(P, "generate_plan",
                               return_value=json.dumps(plan_of(), ensure_ascii=False)), \
                mock.patch.object(G, "generate_sql", side_effect=fake_sql), \
                mock.patch.object(G, "run_sql",
                                  return_value={"ok": True, "sql": GOOD_SQL, "rows": 2,
                                                "df": pd.DataFrame({"区域": ["华东"], "销售额": [1.0]})}):
            state = agent.run("只看华东 6 月的销售额是多少？")

        self.assertIn("SUM(amount)", captured["plan_hint"])
        self.assertTrue(state["plan_used"])
        self.assertEqual(state["plan"]["metric"]["column"], "amount")

    def test_simple_question_does_not_call_plan(self):
        """简单问题一次计划调用都不能有 —— 这是成本红线。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        with mock.patch.object(P, "generate_plan") as plan_call, \
                mock.patch.object(G, "generate_sql", return_value=GOOD_SQL), \
                mock.patch.object(G, "run_sql",
                                  return_value={"ok": True, "sql": GOOD_SQL, "rows": 5,
                                                "df": pd.DataFrame({"区域": ["华东"] * 5,
                                                                    "销售额": [1.0] * 5})}):
            state = agent.run("各区域的销售额是多少？")

        plan_call.assert_not_called()
        self.assertFalse(state.get("plan_used") or False)
        self.assertIn("简单", state.get("plan_reason", ""))

    def test_broken_plan_falls_back(self):
        """模型输出解析不了时，按普通方式生成 —— 计划不能成为单点故障。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        with mock.patch.object(P, "generate_plan", return_value="我不会做计划"), \
                mock.patch.object(G, "generate_sql", return_value=GOOD_SQL) as sql, \
                mock.patch.object(G, "run_sql",
                                  return_value={"ok": True, "sql": GOOD_SQL, "rows": 5,
                                                "df": pd.DataFrame({"区域": ["华东"] * 5,
                                                                    "销售额": [1.0] * 5})}):
            state = agent.run("只看华东 6 月的销售额是多少？")

        sql.assert_called()
        self.assertIsNotNone(state["result"])
        self.assertFalse(state["plan_used"])
        self.assertTrue(any(i["kind"] == "parse" for i in state["plan_issues"]))

    def test_linear_engine_shares_the_same_rules(self):
        """线性编排走同一个 build_plan，不能只有状态机有计划（P2-2 的教训）。"""
        captured = {}

        def fake_sql(schema, terms, question, previous_error=None, previous_empty=False,
                     lessons=None, plan_hint=""):
            captured["plan_hint"] = plan_hint
            return GOOD_SQL

        agent = A.DataAgent(with_chart=False, with_insight=False)
        with mock.patch.object(P, "generate_plan",
                               return_value=json.dumps(plan_of(), ensure_ascii=False)), \
                mock.patch.object(A, "generate_sql", side_effect=fake_sql), \
                mock.patch.object(A, "run_sql",
                                  return_value={"ok": True, "sql": GOOD_SQL, "rows": 2,
                                                "df": pd.DataFrame({"区域": ["华东"], "销售额": [1.0]})}):
            res = agent.ask("只看华东 6 月的销售额是多少？")

        self.assertIn("SUM(amount)", captured["plan_hint"])
        self.assertIsNotNone(res.plan)
        self.assertIsNotNone(res.df)

    def test_linear_engine_blocks_on_fatal(self):
        plan = plan_of(filters=[{"column": "category", "values": ["电脑"], "op": "in"}])
        agent = A.DataAgent(with_chart=False, with_insight=False)
        with mock.patch.object(P, "generate_plan",
                               return_value=json.dumps(plan, ensure_ascii=False)), \
                mock.patch.object(A, "generate_sql") as sql:
            res = agent.ask("只看电脑的销售额")

        sql.assert_not_called()
        self.assertIsNone(res.df)
        self.assertIn("电脑", res.refuse_reason)
        self.assertIn("笔记本", res.refuse_reason)


if __name__ == "__main__":
    unittest.main()
