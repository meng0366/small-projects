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
SPAN_QUESTION = "为什么 2025 年 3-5 月的销售额比 1-2 月高？"


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

    def test_month_span_is_one_anchor(self):
        """「3-5 月」是一个区间锚点，里面的 5 月不再单独算一条。"""
        anchors = extract_anchors("为什么 2025 年 3-5 月的销售额比 1-2 月高？", _enums())
        self.assertEqual(anchors["spans"], [(3, 5), (1, 2)])
        self.assertEqual(anchors["months"], [])
        self.assertEqual(anchors["years"], ["2025"])

    def test_expanded_span_is_fine(self):
        sql = ("SELECT 1 FROM orders WHERE strftime('%Y-%m', order_date) IN "
               "('2025-03','2025-04','2025-05','2025-01','2025-02')")
        self.assertEqual(check_scope(SPAN_QUESTION, sql, _enums()), [])

    def test_span_as_date_range_is_fine(self):
        """上界写成「下一个月 1 日」是常见写法，此时末月不以 -05 出现，不能判成少了 5 月。"""
        sql = ("SELECT 1 FROM orders WHERE order_date >= '2025-03-01' "
               "AND order_date < '2025-06-01' "
               "AND order_date >= '2025-01-01' AND order_date < '2025-03-01'")
        self.assertEqual(check_scope(SPAN_QUESTION, sql, _enums()), [])

    def test_narrowed_span_is_caught(self):
        """把 3-5 月缩成 5 月：拿到了数据也不是用户问的那一段。"""
        sql = "SELECT 1 FROM orders WHERE strftime('%Y-%m', order_date) = '2025-05'"
        problems = check_scope(SPAN_QUESTION, sql, _enums())
        self.assertTrue(any("3-5" in p for p in problems), problems)

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


class TestQuarterAnchors(unittest.TestCase):
    """季度也要守（F3.5 / 期次跨度）。

    季度是有歧义的时段写法：用户写「Q2」，模型可能写成 `2025Q2`，也可能展开成
    `'2025-04'/'2025-05'/'2025-06'` 或日期区间 `>= '2025-04-01' AND < '2025-07-01'`。
    所以不能只按字面守 —— 三种写法都得放行，只拦"换成别的时间段"。
    """

    QUESTION = "为什么 2025 年 Q2 的销售额比 Q1 高？"
    # 逐月展开：Q1+Q2 都在
    SQL_MONTHS = ("SELECT strftime('%Y-%m', order_date) AS ym, SUM(amount) AS amt FROM orders "
                  "WHERE strftime('%Y-%m', order_date) IN "
                  "('2025-01','2025-02','2025-03','2025-04','2025-05','2025-06') GROUP BY ym")
    # 日期区间：上界写成下季度首月 1 日，末月不会以 '-06' 出现
    SQL_RANGE = ("SELECT 1 FROM orders WHERE (order_date >= '2025-01-01' AND order_date < '2025-04-01') "
                 "OR (order_date >= '2025-04-01' AND order_date < '2025-07-01')")
    # 算出季度列再筛，字面是 '2025-Q2'
    SQL_LITERAL = ("SELECT 1 FROM orders WHERE strftime('%Y', order_date) || '-Q' || "
                   "CAST((CAST(strftime('%m', order_date) AS INTEGER) + 2) / 3 AS TEXT) "
                   "IN ('2025-Q1','2025-Q2')")

    def test_quarter_extracted_in_order(self):
        anchors = extract_anchors(self.QUESTION, _enums())
        # 年份只跟紧挨着它写的那个季度绑定，后面光秃秃的「Q1」记为不明年份；
        # 年份本身另有 year 锚点兜底（重试语句里必须有 2025）。
        self.assertEqual(anchors["quarters"], [(2025, 2), (None, 1)])
        self.assertEqual(anchors["months"], [], "季度不该再被拆出一堆月份锚点")
        self.assertEqual(anchors["years"], ["2025"])

    def test_chinese_quarter_written_without_year(self):
        anchors = extract_anchors("为什么第二季度的销售额比第一季度高？", _enums())
        self.assertEqual(anchors["quarters"], [(None, 2), (None, 1)])

    def test_expanded_months_is_fine(self):
        self.assertEqual(check_scope(self.QUESTION, self.SQL_MONTHS, _enums()), [])

    def test_date_range_is_fine(self):
        self.assertEqual(check_scope(self.QUESTION, self.SQL_RANGE, _enums()), [])

    def test_quarter_literal_is_fine(self):
        self.assertEqual(check_scope(self.QUESTION, self.SQL_LITERAL, _enums()), [])

    def test_quarter_narrowed_to_a_month_is_caught(self):
        """Q2 被缩成 6 月：拿到的数不是那一季度。"""
        sql = ("SELECT strftime('%Y-%m', order_date) AS ym, SUM(amount) AS amt FROM orders "
               "WHERE strftime('%Y-%m', order_date) IN "
               "('2025-01','2025-02','2025-03','2025-06') GROUP BY ym")
        problems = check_scope(self.QUESTION, sql, _enums())
        self.assertTrue(any("Q2" in p for p in problems), problems)

    def test_quarter_replaced_by_next_quarter_is_caught(self):
        sql = self.SQL_MONTHS.replace("'2025-04','2025-05','2025-06'",
                                      "'2025-07','2025-08','2025-09'")
        problems = check_scope(self.QUESTION, sql, _enums())
        self.assertTrue(any("2025年Q2" in p for p in problems), problems)

    def test_quarter_without_year_is_not_blocked(self):
        """不带年份的季度不该因为语句里写不写 2025 被误拦。"""
        sql = ("SELECT 1 FROM orders WHERE strftime('%Y-%m', order_date) IN "
               "('2025-04','2025-05','2025-06')")
        self.assertEqual(check_scope("Q2 的销售额是多少？", sql, _enums()), [])

    def test_bare_month_numbers_are_fine(self):
        """有的查询不写日期字符串，直接比月份数字，也得认。"""
        sql = ("SELECT SUM(amount) AS amt FROM orders WHERE strftime('%Y', order_date) = '2025' "
               "AND CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 4 AND 6")
        self.assertEqual(check_scope("2025 年 Q2 的销售额是多少？", sql, _enums()), [])

    def test_limit_number_is_not_a_month(self):
        """`LIMIT 6` 里的 6 不算六月，否则缩了范围的查询会被放过去。"""
        sql = "SELECT 1 FROM orders WHERE strftime('%Y', order_date) = '2025' LIMIT 6"
        problems = check_scope("2025 年 6 月的销售额是多少？", sql, _enums())
        self.assertTrue(any("6 月" in p for p in problems), problems)

    def test_quarter_regex_matches_attribution(self):
        """两处季度正则必须认同一批写法，否则守的时段和算的时段会错位。"""
        from app.tools.attribution import QUARTER_RE as ATTR_RE
        from app.tools.scope_guard import _QUARTER_RE
        samples = ["2025Q2", "2025-Q2", "Q2 比 Q1 高", "第二季度", "第 2 季度",
                   "2025年Q3", "从 Q1 到 Q3", "H1 上半年"]
        for text in samples:
            self.assertEqual([m.group(0) for m in _QUARTER_RE.finditer(text)],
                             [m.group(0) for m in ATTR_RE.finditer(text)], text)


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
