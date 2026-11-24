"""离线冒烟测试：不调用 LLM，验证核心模块可用。

运行：python -m unittest discover -s tests -v
"""
import unittest

import pandas as pd

from app.nodes.execute import check_safety, execute_code, UnsafeCodeError
from app.schema.loader import (
    build_schema_text,
    load_catalog_terms,
    get_db_path,
    list_tables,
    build_db_schema_text,
)
from app.tools.sql_tool import run_sql


class TestSafety(unittest.TestCase):
    """代码安全检查（借鉴 OpenChatBI 的"校验前置"）。"""

    def test_blocks_os_import(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety("import os")

    def test_blocks_file_open(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety('result = open("secret.txt").read()')

    def test_blocks_network(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety("import requests\nrequests.get('http://x')")

    def test_blocks_syntax_error(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety("result = df.groupby(")

    def test_allows_safe_code(self):
        check_safety('result = df["amount"].sum()')  # 不应抛异常


class TestExecute(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "region": ["华东", "华南", "华东"],
            "amount": [10.0, 20.0, 30.0],
        })

    def test_groupby_sum(self):
        r = execute_code('result = df.groupby("region")["amount"].sum()', self.df)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(float(r["result"]["华东"]), 40.0)

    def test_missing_result(self):
        r = execute_code("x = 1", self.df)
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "result")

    def test_runtime_error(self):
        r = execute_code('result = df["not_exist"].sum()', self.df)
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "runtime")

    def test_original_df_not_modified(self):
        execute_code('df["new_col"] = 1\nresult = 1', self.df)
        self.assertNotIn("new_col", self.df.columns)


class TestSchema(unittest.TestCase):
    def test_schema_text_contains_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        text = build_schema_text(df)
        self.assertIn("a", text)
        self.assertIn("b", text)

    def test_catalog_terms_is_text(self):
        self.assertIsInstance(load_catalog_terms(), str)


class TestDbSchema(unittest.TestCase):
    """SQLite 多表接入与 Schema 描述（F1.2 / F1.3）。"""

    def setUp(self):
        self.db_path = get_db_path()

    def test_list_tables(self):
        tables = list_tables(self.db_path)
        for expected in ("orders", "customers", "products"):
            self.assertIn(expected, tables)

    def test_db_schema_has_relations_and_enums(self):
        text = build_db_schema_text(self.db_path)
        self.assertIn("表关系", text)
        self.assertIn("orders.customer_id -> customers.customer_id", text)
        # 枚举值必须显式给出，避免模型臆造字面量
        self.assertIn("华东", text)
        self.assertIn("笔记本", text)

    def test_db_schema_mentions_row_count(self):
        text = build_db_schema_text(self.db_path)
        self.assertIn("3877 行", text)


class TestSqlTool(unittest.TestCase):
    """SQL 安全校验与只读执行（F2.3 / F2.4）。"""

    def setUp(self):
        self.db_path = get_db_path()

    def test_select_runs(self):
        r = run_sql("SELECT region, COUNT(*) AS n FROM customers GROUP BY region",
                    self.db_path)
        self.assertTrue(r["ok"])
        self.assertEqual(r["rows"], 5)

    def test_blocks_delete(self):
        r = run_sql("DELETE FROM orders", self.db_path)
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "safety")

    def test_blocks_multiple_statements(self):
        r = run_sql("SELECT 1; DROP TABLE orders", self.db_path)
        self.assertFalse(r["ok"])

    def test_blocks_comment_bypass(self):
        r = run_sql("SELECT 1 /* x */; DELETE FROM orders", self.db_path)
        self.assertFalse(r["ok"])

    def test_blocks_pragma(self):
        r = run_sql("PRAGMA table_info(orders)", self.db_path)
        self.assertFalse(r["ok"])

    def test_with_delete_blocked(self):
        r = run_sql("WITH t AS (SELECT 1) DELETE FROM orders", self.db_path)
        self.assertFalse(r["ok"])

    def test_truncation(self):
        r = run_sql("SELECT * FROM orders", self.db_path, max_rows=10)
        self.assertTrue(r["ok"])
        self.assertEqual(r["rows"], 10)
        self.assertTrue(r["truncated"])

    def test_runtime_error_reported(self):
        r = run_sql("SELECT not_a_column FROM orders", self.db_path)
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "runtime")


if __name__ == "__main__":
    unittest.main()
