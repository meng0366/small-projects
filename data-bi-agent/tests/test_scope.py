"""范围守卫的测试（P0-2b）：空结果自愈不许把用户的条件换掉。

为什么单独一组用例：这不是"体验优化"，而是**防止给出错误数据**。
真实评测里模型把「2024 年 3 月」在重试时改成「2025-03」，拿回 100 行真实数据，
回答里一个字都不提 —— 用户不会怀疑，因为格式、数字、列名全是对的。
这类错误靠提示词堵不住（提示词原话就是"放宽范围"），必须用代码兜住。
"""
import unittest
from unittest import mock

import pandas as pd

import app.graph as G
from app.tools.scope_guard import (check_scope, extract_anchors, looks_like_sql,
                                   parse_defer, strip_code_fence)

SQL_2024 = ("SELECT * FROM orders WHERE strftime('%Y-%m', order_date) = '2024-03' LIMIT 100")
SQL_2025 = ("SELECT * FROM orders WHERE strftime('%Y-%m', order_date) = '2025-03' LIMIT 100")
QUESTION = "2024 年 3 月的订单明细有哪些？"


def _enums():
    from app.context import AgentContext
    return AgentContext().enums


class TestScopeGuard(unittest.TestCase):
    def test_extract_anchors(self):
        anchors = extract_anchors("2024 年 3 月华东的手机销售额", _enums())
        self.assertEqual(anchors["years"], ["2024"])
        self.assertEqual(anchors["months"], [3])
        self.assertEqual([v["value"] for v in anchors["values"]], ["华东", "手机"])

    def test_no_anchor_no_problem(self):
        """没提年份/取值的口语问题不该被拦。"""
        self.assertEqual(check_scope("哪个片区卖得最好？", "SELECT 1", _enums()), [])

    def test_year_changed_is_caught(self):
        problems = check_scope(QUESTION, SQL_2025, _enums())
        self.assertEqual(len(problems), 1)
        self.assertIn("2024", problems[0])

    def test_year_kept_is_fine(self):
        self.assertEqual(check_scope(QUESTION, SQL_2024, _enums()), [])

    def test_month_range_still_fine(self):
        """同一个月换成日期区间是正常写法，不能误伤。"""
        sql = ("SELECT * FROM orders WHERE order_date >= '2024-03-01' "
               "AND order_date < '2024-04-01'")
        self.assertEqual(check_scope(QUESTION, sql, _enums()), [])

    def test_dropped_enum_value_is_caught(self):
        problems = check_scope("华东区域的销售额是多少？",
                               "SELECT SUM(amount) FROM orders", _enums())
        self.assertEqual(len(problems), 1)
        self.assertIn("华东", problems[0])

    def test_sql_detection(self):
        self.assertTrue(looks_like_sql("SELECT 1"))
        self.assertTrue(looks_like_sql("```sql\nWITH t AS (SELECT 1) SELECT * FROM t\n```"))
        self.assertTrue(looks_like_sql("-- 说明注释\nSELECT 1"))
        self.assertFalse(looks_like_sql("数据库中不存在退货相关字段，无法计算退货率。"))

    def test_parse_defer(self):
        self.assertEqual(parse_defer("DEFER: 缺少成本字段"), "缺少成本字段")
        self.assertEqual(parse_defer("```\nDEFER：没有退货表\n```"), "没有退货表")
        self.assertIsNone(parse_defer("SELECT 1"))
        self.assertIsNone(parse_defer("defers 不是关键字"))

    def test_strip_code_fence(self):
        self.assertEqual(strip_code_fence("```sql\nSELECT 1\n```"), "SELECT 1")
        self.assertEqual(strip_code_fence("SELECT 1"), "SELECT 1")


class TestScopeGuardInGraph(unittest.TestCase):
    """在状态机里验证：改条件的重试结果必须被丢掉。"""

    def test_changed_year_result_is_rejected(self):
        frames = [pd.DataFrame(), pd.DataFrame({"order_id": ["O1"], "order_date": ["2025-03-01"]})]
        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=2)
        call = {"n": 0}

        def fake_run_sql(statement, db_path):
            frame = frames[min(call["n"], 1)]
            call["n"] += 1
            return {"ok": True, "df": frame, "sql": statement, "rows": len(frame)}

        with mock.patch.object(G, "generate_sql", side_effect=[SQL_2024, SQL_2025]):
            with mock.patch.object(G, "run_sql", side_effect=fake_run_sql):
                state = agent.run(QUESTION)

        self.assertIsNone(state.get("result"), "改条件换来的数据必须丢掉")
        self.assertTrue(state.get("scope_violation"))
        self.assertIn("2024", state["answer"])
        self.assertIn("没有拿别的范围顶替", state["answer"])
        # 不渲染空图空表
        self.assertEqual(state["chart_type"], "none")

    def test_same_scope_empty_is_not_flagged(self):
        """两轮都是空：按普通空结果处理，不要误报成"改了条件"。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=2)
        with mock.patch.object(G, "generate_sql", return_value=SQL_2024):
            with mock.patch.object(G, "run_sql",
                                   return_value={"ok": True, "df": pd.DataFrame(),
                                                 "sql": SQL_2024, "rows": 0}):
                state = agent.run(QUESTION)
        self.assertIsNone(state.get("scope_violation"))
        self.assertIn("没有匹配到数据", state["answer"])

    def test_model_declares_defer(self):
        """模型直接说缺字段时，不要拿去执行 SQL。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=2)
        with mock.patch.object(G, "generate_sql", return_value="DEFER: 缺少成本字段，无法计算毛利率"):
            with mock.patch.object(G, "run_sql") as run_sql:
                state = agent.run("每个商品的毛利率是多少？")
        run_sql.assert_not_called()
        self.assertIn("缺少成本字段", state["answer"])
        self.assertIsNone(state.get("result"))
        self.assertEqual(state["retry_count"], 0, "已判定答不了，不该再重试")

    def test_prose_statement_is_treated_as_refusal(self):
        """整段自然语言（不是 SQL）也要按拒答处理，不能丢给执行器报错。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=2)
        prose = "数据库中不存在退货相关字段，无法计算退货率。"
        with mock.patch.object(G, "generate_sql", return_value=prose):
            with mock.patch.object(G, "run_sql") as run_sql:
                state = agent.run("各商品的退货率是多少？")
        run_sql.assert_not_called()
        self.assertIn("退货", state["answer"])


if __name__ == "__main__":
    unittest.main()
