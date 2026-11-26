"""SQL 结构审查的离线测试：纯函数判据 + 两引擎接入 + 输出结构。

覆盖第十轮 P1 的三类问题：笛卡尔积（拦）、SELECT * 与非唯一键关联（提示），
以及最容易踩的误报：字符串/注释里的关键字、COUNT(*)、CTE 子查询、评测金标准。
全部 mock LLM 或纯函数，不花 token。
"""
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

try:
    import _env  # noqa: F401  测试环境隔离（临时 AGENT_HOME、关闭 SQL 复用）
except ModuleNotFoundError:      # 以包方式运行（python -m unittest tests.test_xxx）
    from . import _env  # noqa: F401

import yaml

import app.agent as A
import app.graph as G
from app.payload import build_payload
from app.schema.loader import describe_unique_keys
from app.tools.sql_review import (
    BLOCK, WARN, format_review_for_model, format_review_note, review_sql,
)

GOOD_SQL = ("SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额 "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.region")
CARTESIAN = "SELECT o.amount FROM orders o, customers c"
KEYS = {"orders": set(), "customers": {"customer_id"}, "products": {"product_id"}}


def codes(sql, keys=KEYS):
    return [item["code"] for item in review_sql(sql, keys)["issues"]]


class TestReviewRules(unittest.TestCase):
    def test_select_star(self):
        self.assertEqual(codes("SELECT * FROM orders"), ["select_star"])
        self.assertEqual(codes("SELECT o.* FROM orders o"), ["select_star"])
        self.assertEqual(codes("SELECT DISTINCT * FROM orders"), ["select_star"])

    def test_count_star_is_not_select_star(self):
        self.assertEqual(codes("SELECT COUNT(*) FROM orders"), [])
        self.assertEqual(codes("SELECT COUNT(DISTINCT o.id) FROM orders o"), [])

    def test_string_and_comment_do_not_count(self):
        sql = "SELECT o.amount FROM orders o -- SELECT * FROM fake\n WHERE 'SELECT *' = 'x'"
        self.assertEqual(codes(sql), [])

    def test_comma_join_without_condition_is_blocked(self):
        result = review_sql(CARTESIAN, KEYS)
        self.assertFalse(result["ok"])
        self.assertEqual(result["blocks"], 1)
        self.assertEqual(result["issues"][0]["level"], BLOCK)

    def test_comma_join_with_condition_passes(self):
        self.assertEqual(codes(CARTESIAN + " WHERE o.customer_id = c.customer_id"), [])

    def test_join_without_on_is_blocked(self):
        self.assertFalse(review_sql("SELECT o.amount FROM orders o JOIN customers c")["ok"])
        self.assertEqual(codes("SELECT o.amount FROM orders o JOIN customers c"),
                         ["join_no_on"])

    def test_join_with_using_passes(self):
        self.assertEqual(codes("SELECT o.amount FROM orders o JOIN customers c "
                               "USING (customer_id)"), [])

    def test_always_true_on_is_blocked(self):
        self.assertEqual(codes("SELECT o.amount FROM orders o JOIN customers c ON 1=1"),
                         ["join_always_true"])
        self.assertEqual(codes("SELECT o.amount FROM orders o JOIN customers c ON TRUE"),
                         ["join_always_true"])

    def test_explicit_cross_join_is_warn_only(self):
        result = review_sql("SELECT o.amount FROM orders o CROSS JOIN customers c", KEYS)
        self.assertTrue(result["ok"])
        self.assertEqual(result["issues"][0]["level"], WARN)

    def test_non_unique_join_warns(self):
        result = review_sql("SELECT o.amount FROM orders o JOIN orders o2 "
                            "ON o.customer_id = o2.customer_id", KEYS)
        self.assertTrue(result["ok"])       # warn 不拦执行
        self.assertEqual([i["code"] for i in result["issues"]], ["non_unique_join"])

    def test_one_unique_side_passes(self):
        self.assertEqual(codes(GOOD_SQL, KEYS), [])

    def test_unknown_uniqueness_does_not_warn(self):
        """拿不到唯一性信息时宁可不报：不猜"看起来像 ID 的列就是唯一的"。"""
        self.assertEqual(review_sql("SELECT o.amount FROM orders o JOIN orders o2 "
                                    "ON o.customer_id = o2.customer_id")["issues"], [])

    def test_cte_subquery_not_false_positive(self):
        sql = ("WITH t AS (SELECT * FROM orders o JOIN customers c "
               "ON o.customer_id = c.customer_id) SELECT t.amount FROM t")
        self.assertEqual(codes(sql, KEYS), [])

    def test_all_gold_sql_are_clean(self):
        """评测金标准是一条硬基线：审查器升级后也不能给它们报问题。"""
        path = os.path.join(os.path.dirname(__file__), "..", "evals", "cases.yaml")
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cases = data.get("cases", data if isinstance(data, list) else [])
        checked = 0
        for case in cases:
            sql = case.get("gold_sql")
            if not sql:
                continue
            checked += 1
            self.assertEqual(codes(sql, KEYS), [], f"金标准被误报：{case.get('id')}")
        self.assertGreater(checked, 30)

    def test_format_helpers(self):
        block = review_sql(CARTESIAN, KEYS)["issues"]
        self.assertIn("笛卡尔积", format_review_for_model(block))
        self.assertEqual(format_review_note(block), "")   # block 不展示给用户提示
        warn = review_sql("SELECT * FROM orders", KEYS)["issues"]
        self.assertIn("SELECT *", format_review_note(warn))
        self.assertEqual(format_review_for_model(warn), "")

    def test_empty_sql_is_clean(self):
        result = review_sql("", KEYS)
        self.assertTrue(result["ok"])
        self.assertEqual(result["issues"], [])


class TestUniqueKeys(unittest.TestCase):
    def test_pragmas_and_catalog_declarations(self):
        with tempfile.TemporaryDirectory(prefix="sql-review-") as folder:
            path = os.path.join(folder, "t.db")
            con = sqlite3.connect(path)
            try:
                con.execute("CREATE TABLE dim (id TEXT PRIMARY KEY, label TEXT)")
                con.execute("CREATE TABLE code (code TEXT UNIQUE, name TEXT)")
                con.execute("CREATE TABLE fact (id TEXT, dim_id TEXT, amount REAL)")
                con.commit()
            finally:
                con.close()
            catalog = {
                "tables": [{"name": "fact", "columns": {"other": {"ref": "code.code"}}}],
                "relationships": ["fact.dim_id -> dim.id (多对一)"],
            }
            keys = describe_unique_keys(path, catalog)
            self.assertIn("id", keys["dim"])
            self.assertIn("code", keys["code"])      # UNIQUE 索引
            self.assertIn("code", keys["code"])      # catalog ref 同样指向它
            self.assertIn("id", keys["dim"])         # 多对一声明
            self.assertEqual(keys.get("fact", set()), set())

    def test_composite_primary_key_not_counted_as_single_unique(self):
        with tempfile.TemporaryDirectory(prefix="sql-review-") as folder:
            path = os.path.join(folder, "t.db")
            con = sqlite3.connect(path)
            try:
                con.execute("CREATE TABLE pair (a TEXT, b TEXT, PRIMARY KEY (a, b))")
                con.commit()
            finally:
                con.close()
            self.assertEqual(describe_unique_keys(path, {}).get("pair", set()), set())


class TestOutput(unittest.TestCase):
    def test_payload_exposes_review(self):
        state = {"review_issues": [{"code": "select_star", "level": WARN,
                                    "message": "提示"}],
                 "review_blocked": False}
        review = build_payload(state, "q")["review"]
        self.assertFalse(review["blocked"])
        self.assertEqual(review["issues"][0]["code"], "select_star")


class TestGraphIntegration(unittest.TestCase):
    def _agent(self, max_retries=2):
        return G.GraphAgent(with_chart=False, with_insight=False, max_retries=max_retries)

    def test_blocked_sql_retries_staying_on_sql_route(self):
        seen = []

        def generate(schema, terms, question, previous_error=None, previous_empty=False,
                     lessons=None, plan_hint=""):
            seen.append(previous_error)
            return CARTESIAN if len(seen) == 1 else GOOD_SQL

        agent = self._agent()
        with mock.patch.object(G, "generate_sql", side_effect=generate):
            state = agent.run("各区域的销售额是多少？")
        self.assertEqual(state["route"], "sql")     # 没被换到 pandas
        self.assertEqual(state["retry_count"], 1)
        self.assertIsNotNone(state["result"])
        self.assertIn("结构审查", seen[1])           # 审查意见回喂给模型
        self.assertIn("结构审查", state["errors"][0])
        self.assertEqual(state["review_issues"], [])  # 新语句没有旧提示

    def test_blocked_sql_without_budget_reports_reason(self):
        agent = self._agent(max_retries=0)
        with mock.patch.object(G, "generate_sql", return_value=CARTESIAN):
            state = agent.run("各区域的销售额是多少？")
        self.assertIsNone(state["result"])
        self.assertIn("结构审查", state["answer"])
        self.assertTrue(state["review_blocked"])

    def test_warn_only_still_executes(self):
        agent = self._agent()
        with mock.patch.object(G, "generate_sql", return_value="SELECT * FROM orders"):
            state = agent.run("各区域的销售额是多少？")
        self.assertIsNotNone(state["result"])
        self.assertEqual(state["retry_count"], 0)
        self.assertEqual([i["code"] for i in state["review_issues"]], ["select_star"])

    def test_review_can_be_disabled(self):
        agent = self._agent()
        agent.ctx.use_review = False
        with mock.patch.object(G, "generate_sql", return_value=CARTESIAN):
            state = agent.run("各区域的销售额是多少？")
        self.assertIsNotNone(state["result"])
        self.assertEqual(state["review_issues"], [])


class TestLinearIntegration(unittest.TestCase):
    def test_blocked_sql_regenerates_once_and_keeps_sql_route(self):
        agent = A.DataAgent(with_chart=False, with_insight=False)
        with mock.patch.object(A, "generate_sql", side_effect=[CARTESIAN, GOOD_SQL]) as gen:
            res = agent.ask("各区域的销售额是多少？")
        self.assertEqual(gen.call_count, 2)
        self.assertEqual(res.route, "sql")
        self.assertIsNotNone(res.df)
        self.assertTrue(any("结构审查" in e for e in res.errors))
        self.assertEqual(res.review_issues, [])

    def test_blocked_twice_does_not_fallback_to_pandas(self):
        agent = A.DataAgent(with_chart=False, with_insight=False)
        with mock.patch.object(A, "generate_sql", return_value=CARTESIAN):
            with mock.patch.object(A, "generate_code") as code_gen:
                res = agent.ask("各区域的销售额是多少？")
        self.assertIsNone(res.df)
        self.assertTrue(res.review_blocked)
        self.assertEqual(code_gen.call_count, 0)   # 跨表题不做单表近似答案


if __name__ == "__main__":
    unittest.main()
