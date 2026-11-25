"""两段式探查（P1）的离线测试：不调模型，真库只读跑 SQL。

覆盖：
- 解析：`PEEK:` 前缀的各种写法、正文夹带、二次探查
- 白名单：合法形态放行；写操作 / 多语句 / JOIN / UNION / 子查询 / 聚合 / `*` / 多表全部拒绝
- 回灌：单列取值清单、多列按行、空值标注、超长截断、行数截断、失败原因
- 额度：每次问答一次；用完仍会重新生成一次并明说；二次探查被剥掉，不会死循环
- 两引擎：状态机与线性编排都走同一条 `maybe_peek`
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from app.context import AgentContext
from app.schema.loader import get_db_path
from app.tools.peek import (
    MAX_PEEKS,
    PEEK_MAX_ROWS,
    PEEK_MAX_VALUES,
    PEEK_OFF_NOTE,
    PeekError,
    check_peek_sql,
    format_peek_result,
    maybe_peek,
    parse_peek,
    run_peek,
    strip_peek,
    summarize_peek,
)

PEEK_SQL = "PEEK: SELECT DISTINCT region FROM customers"
GOOD_SQL = ("SELECT c.region AS 区域, COUNT(*) AS 客户数 FROM customers c "
            "GROUP BY c.region ORDER BY 客户数 DESC")


def fake_ctx(use_peek: bool = True, db_path=None) -> SimpleNamespace:
    """最小上下文：maybe_peek 只用到 use_peek 与 db_path。"""
    return SimpleNamespace(use_peek=use_peek,
                           db_path=db_path if db_path is not None else get_db_path())


class TestParsePeek(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(parse_peek("PEEK: SELECT DISTINCT category FROM products"),
                         "SELECT DISTINCT category FROM products")

    def test_fullwidth_colon(self):
        """中文输入法下模型常写全角冒号，认不出来就等于功能不存在。"""
        self.assertEqual(parse_peek("PEEK：SELECT DISTINCT region FROM customers"),
                         "SELECT DISTINCT region FROM customers")

    def test_case_insensitive(self):
        self.assertEqual(parse_peek("peek: SELECT 1 FROM orders"), "SELECT 1 FROM orders")

    def test_trailing_semicolon_removed(self):
        self.assertEqual(parse_peek("PEEK: SELECT DISTINCT region FROM customers;"),
                         "SELECT DISTINCT region FROM customers")

    def test_none_when_absent(self):
        self.assertIsNone(parse_peek("SELECT DISTINCT region FROM customers"))

    def test_takes_first_of_multiple(self):
        text = "PEEK: SELECT DISTINCT a FROM t1\nPEEK: SELECT DISTINCT b FROM t2"
        self.assertEqual(parse_peek(text), "SELECT DISTINCT a FROM t1")

    def test_empty_peek_is_none(self):
        self.assertIsNone(parse_peek("PEEK:   "))

    def test_strip_removes_peek_line(self):
        text = "PEEK: SELECT DISTINCT region FROM customers\nSELECT 1 FROM orders"
        self.assertEqual(strip_peek(text), "SELECT 1 FROM orders")

    def test_strip_leaves_plain_sql_untouched(self):
        self.assertEqual(strip_peek(GOOD_SQL), GOOD_SQL)


class TestCheckPeekSql(unittest.TestCase):
    def test_accepts_distinct_column(self):
        sql = "SELECT DISTINCT category FROM products"
        self.assertEqual(check_peek_sql(sql), sql)

    def test_accepts_where_order_limit(self):
        sql = "SELECT DISTINCT region FROM customers WHERE region != 'x' ORDER BY region LIMIT 10"
        self.assertEqual(check_peek_sql(sql), sql)

    def test_accepts_table_alias(self):
        sql = "SELECT DISTINCT c.region FROM customers c"
        self.assertEqual(check_peek_sql(sql), sql)

    def test_accepts_multiple_columns(self):
        sql = "SELECT DISTINCT region, customer_level FROM customers"
        self.assertEqual(check_peek_sql(sql), sql)

    def test_rejects_write_statement(self):
        with self.assertRaises(PeekError):
            check_peek_sql("DELETE FROM orders")

    def test_rejects_multiple_statements(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT region FROM customers; SELECT 1")

    def test_rejects_join(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT c.region FROM customers c "
                           "JOIN orders o ON o.customer_id = c.customer_id")

    def test_rejects_union(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT region FROM customers "
                           "UNION SELECT DISTINCT category FROM products")

    def test_rejects_subquery(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT region FROM customers "
                           "WHERE customer_id IN (SELECT customer_id FROM orders)")

    def test_rejects_with(self):
        with self.assertRaises(PeekError):
            check_peek_sql("WITH t AS (SELECT 1) SELECT DISTINCT region FROM customers")

    def test_rejects_aggregate(self):
        """探查是"看一眼取值"，不是跑统计；聚合一律不放行。"""
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT COUNT(*) FROM orders")

    def test_rejects_star(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT * FROM customers")

    def test_rejects_too_many_columns(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT a, b, c, d, e FROM t")

    def test_rejects_expression_column(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT strftime('%Y', order_date) FROM orders")

    def test_rejects_multiple_tables_by_comma(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT region FROM customers, orders")

    def test_rejects_without_from(self):
        with self.assertRaises(PeekError):
            check_peek_sql("SELECT DISTINCT 1")


class TestFormat(unittest.TestCase):
    def _single(self, values, truncated=False, table="products", column="category"):
        import pandas as pd
        return {"ok": True, "df": pd.DataFrame({column: values}), "table": table,
                "columns": [column], "rows": len(values), "truncated": truncated,
                "error": ""}

    def test_single_column_values(self):
        result = self._single(["手机", "笔记本", "平板", "笔记本"])
        text = format_peek_result(result)
        self.assertIn("category 共 3 个不同取值", text)
        self.assertIn("手机 / 笔记本 / 平板", text)
        self.assertIn("请据此写出最终 SQL", text)

    def test_values_truncated(self):
        values = [f"v{i}" for i in range(PEEK_MAX_VALUES + 5)]
        text = format_peek_result(self._single(values))
        self.assertIn(f"共 {PEEK_MAX_VALUES + 5} 个不同取值", text)
        self.assertIn("还有 5 个未显示", text)
        self.assertNotIn(f"v{PEEK_MAX_VALUES + 4}", text)

    def test_null_marked(self):
        text = format_peek_result(self._single(["A", None, "B"]))
        self.assertIn("另有 1 行为空", text)

    def test_all_null(self):
        text = format_peek_result(self._single([None, None]))
        self.assertIn("(全部为空)", text)

    def test_empty_result(self):
        text = format_peek_result(self._single([]))
        self.assertIn("没有任何数据", text)

    def test_truncated_notes_partial(self):
        text = format_peek_result(self._single(["A", "B"], truncated=True))
        self.assertIn(f"只看了前 {PEEK_MAX_ROWS} 行", text)

    def test_multi_column_rows(self):
        import pandas as pd
        result = {"ok": True, "table": "customers",
                  "df": pd.DataFrame({"region": ["华东", "华北"],
                                      "customer_level": ["A", "B"]}),
                  "columns": ["region", "customer_level"], "rows": 2,
                  "truncated": False, "error": ""}
        text = format_peek_result(result)
        self.assertIn("customers(region, customer_level)", text)
        self.assertIn("华东 | A", text)

    def test_long_value_truncated(self):
        text = format_peek_result(self._single(["x" * 100]))
        self.assertIn("…", text)
        self.assertNotIn("x" * 100, text)

    def test_failure_message(self):
        text = format_peek_result({"ok": False, "error": "no such column: nope",
                                   "df": None, "table": "orders", "columns": [],
                                   "rows": 0, "truncated": False})
        self.assertIn("探查未成功", text)
        self.assertIn("no such column", text)
        self.assertIn("不要再输出 PEEK", text)

    def test_summarize_single_column(self):
        note = summarize_peek(self._single(["手机", "笔记本"]))
        self.assertIn("products.category → 2 个取值", note)
        self.assertIn("手机", note)

    def test_summarize_failure(self):
        note = summarize_peek({"ok": False, "error": "拒绝执行", "df": None,
                               "table": "", "columns": []})
        self.assertIn("未成功", note)

    def test_summarize_multi_column(self):
        import pandas as pd
        result = {"ok": True, "table": "customers",
                  "df": pd.DataFrame({"region": ["华东"], "customer_level": ["A"]}),
                  "columns": ["region", "customer_level"], "rows": 1,
                  "truncated": False, "error": ""}
        self.assertIn("1 行", summarize_peek(result))


class TestRunPeek(unittest.TestCase):
    def test_runs_real_query(self):
        result = run_peek("SELECT DISTINCT category FROM products")
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(result["table"], "products")
        self.assertGreater(result["rows"], 0)
        self.assertIn("category", result["columns"])

    def test_bad_column_reports_error(self):
        """列名猜错是最常见的情况，错误必须能回灌给模型（它据此改正）。"""
        result = run_peek("SELECT DISTINCT nope FROM products")
        self.assertFalse(result["ok"])
        self.assertIn("nope", result["error"])

    def test_rejected_sql_not_executed(self):
        result = run_peek("PEEK: DELETE FROM orders")   # 带前缀也不该被当 SQL
        self.assertFalse(result["ok"])

    def test_row_limit(self):
        result = run_peek("SELECT DISTINCT order_date FROM orders")
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], PEEK_MAX_ROWS)
        self.assertTrue(result["truncated"])


class TestMaybePeek(unittest.TestCase):
    def test_plain_sql_untouched(self):
        text, update = maybe_peek(fake_ctx(), GOOD_SQL, lambda h: "不该被调用")
        self.assertEqual(text, GOOD_SQL)
        self.assertFalse(update["peek_used"])

    def test_peek_then_regenerate(self):
        hints = []

        def regen(hint):
            hints.append(hint)
            return GOOD_SQL

        text, update = maybe_peek(fake_ctx(), PEEK_SQL, regen)
        self.assertEqual(text, GOOD_SQL)
        self.assertTrue(update["peek_used"])
        self.assertGreater(update["peek_rows"], 0)
        self.assertIn("探查结果", hints[0])
        self.assertIn("华东", hints[0])          # 真实取值回灌进去了
        self.assertIn("华东", update["peek_note"])
        self.assertGreaterEqual(update["peek_ms"], 0)

    def test_rejected_peek_still_regenerates_with_reason(self):
        hints = []

        def regen(hint):
            hints.append(hint)
            return GOOD_SQL

        _, update = maybe_peek(fake_ctx(), "PEEK: DELETE FROM orders", regen)
        self.assertTrue(update["peek_used"])
        self.assertTrue(update["peek_error"])
        self.assertIn("探查未成功", hints[0])
        self.assertEqual(update["peek_rows"], 0)

    def test_remaining_zero_tells_model(self):
        hints = []
        text, update = maybe_peek(fake_ctx(), PEEK_SQL,
                                  lambda h: (hints.append(h), GOOD_SQL)[1],
                                  remaining=0)
        self.assertEqual(text, GOOD_SQL)
        self.assertIn("额度已用完", hints[0])
        self.assertIn("额度已用完", update["peek_error"])
        self.assertEqual(update["peek_rows"], 0)

    def test_second_peek_stripped(self):
        """模型拿到结果还在要探查 → 不执行，也不把 PEEK 行当 SQL 交出去。"""
        text, update = maybe_peek(fake_ctx(), PEEK_SQL,
                                  lambda h: "PEEK: SELECT DISTINCT region FROM customers")
        self.assertEqual(text, "")
        self.assertIn("再探查", update["peek_error"])

    def test_second_peek_with_sql_kept(self):
        text, _ = maybe_peek(fake_ctx(), PEEK_SQL,
                             lambda h: "PEEK: SELECT DISTINCT x FROM customers\n" + GOOD_SQL)
        self.assertEqual(text, GOOD_SQL)

    def test_regenerate_failure_returns_empty(self):
        def boom(hint):
            raise RuntimeError("网络断了")

        text, update = maybe_peek(fake_ctx(), PEEK_SQL, boom)
        self.assertEqual(text, "")
        self.assertIn("重新生成失败", update["peek_error"])

    def test_disabled_strips_peek(self):
        text, update = maybe_peek(fake_ctx(use_peek=False),
                                  PEEK_SQL + "\n" + GOOD_SQL, lambda h: "不该被调用")
        self.assertEqual(text, GOOD_SQL)
        self.assertIn("探查已关闭", update["peek_error"])

    def test_no_db_strips_peek(self):
        ctx = SimpleNamespace(use_peek=True, db_path=None)
        text, update = maybe_peek(ctx, PEEK_SQL + "\n" + GOOD_SQL, lambda h: "不该被调用")
        self.assertEqual(text, GOOD_SQL)
        self.assertIn("不支持探查", update["peek_error"])

    def test_default_quota_is_one(self):
        self.assertEqual(MAX_PEEKS, 1)


class TestEngineWiring(unittest.TestCase):
    """两段式探查必须在两个引擎上都生效（P2-2 的教训：可切换要按接口验证）。"""

    def test_graph_peek_then_sql(self):
        import app.graph as G

        hints = []

        def fake(schema, terms, question, previous_error=None, previous_empty=False,
                 lessons=None, plan_hint=""):
            hints.append(plan_hint)
            return PEEK_SQL if len(hints) == 1 else GOOD_SQL

        ctx = AgentContext(use_plan=False, use_lessons=False)
        agent = G.GraphAgent(ctx=ctx, with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_sql", fake):
            state = agent.run("各区域有多少客户？")

        self.assertTrue(state["peek_used"])
        self.assertGreater(state["peek_rows"], 0)
        self.assertIn("探查结果", hints[1])
        self.assertIsNotNone(state["result"])
        self.assertEqual(len(state["result"]), 5)

    def test_linear_peek_then_sql(self):
        import app.agent as A

        hints = []

        def fake(schema, terms, question, previous_error=None, previous_empty=False,
                 lessons=None, plan_hint=""):
            hints.append(plan_hint)
            return PEEK_SQL if len(hints) == 1 else GOOD_SQL

        ctx = AgentContext(use_plan=False, use_lessons=False)
        agent = A.DataAgent(ctx=ctx)
        with mock.patch.object(A, "generate_sql", fake):
            res = agent.ask("各区域有多少客户？", with_chart=False, with_insight=False)

        self.assertTrue(res.peek.get("peek_used"))
        self.assertIn("探查结果", hints[1])
        self.assertIsNotNone(res.df)
        self.assertEqual(res.route, "sql")

    def test_graph_without_peek_unchanged(self):
        import app.graph as G

        ctx = AgentContext(use_plan=False, use_lessons=False)
        agent = G.GraphAgent(ctx=ctx, with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_sql", return_value=GOOD_SQL):
            state = agent.run("各区域有多少客户？")

        self.assertFalse(state.get("peek_used"))
        self.assertIsNotNone(state["result"])

    def test_peek_off_note_reaches_prompt(self):
        import app.graph as G

        hints = []

        def fake(schema, terms, question, previous_error=None, previous_empty=False,
                 lessons=None, plan_hint=""):
            hints.append(plan_hint)
            return GOOD_SQL

        ctx = AgentContext(use_plan=False, use_lessons=False, use_peek=False)
        agent = G.GraphAgent(ctx=ctx, with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_sql", fake):
            state = agent.run("各区域有多少客户？")

        self.assertIn(PEEK_OFF_NOTE, hints[0])
        self.assertFalse(state.get("peek_used"))


if __name__ == "__main__":
    unittest.main()
