"""安全防护的离线测试（对应 S5.3）：不调用 LLM。

覆盖：
- 静态校验：import 绕过写法、dunder 逃逸、写盘方法、超长代码都要拦住，正常代码不能误伤
- 执行超时：慢代码不能把调用方卡住
- 结果截断：超大结果集不会被原样返回
- SQL 超时：大笛卡尔积/长递归能被中断
- 日志：能落盘成 JSONL，目录不可写时不影响主流程
"""
import importlib
import json
import os
import shutil
import tempfile
import time
import unittest

import pandas as pd

from app.nodes.execute import (
    MAX_CODE_NODES,
    UnsafeCodeError,
    check_safety,
    execute_code,
)
from app.tools.sql_tool import run_sql
from app.schema.loader import get_db_path


class TestStaticCheck(unittest.TestCase):
    """AST 静态校验。"""

    def test_blocks_multi_space_import(self):
        # 纯字符串匹配 `import os` 会被多空格绕过，AST 不受影响
        with self.assertRaises(UnsafeCodeError):
            check_safety("import  os\nresult = 1")

    def test_blocks_from_import(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety("from os import system\nresult = 1")

    def test_blocks_dunder_escape(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety("result = ().__class__.__bases__")

    def test_blocks_eval_name(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety('result = eval("1+1")')

    def test_blocks_getattr_name(self):
        with self.assertRaises(UnsafeCodeError):
            check_safety('result = getattr(df, "shape")')

    def test_blocks_write_to_disk(self):
        for code in ('df.to_csv("out.csv")', 'df.to_excel("out.xlsx")',
                     'df.to_sql("t", None)'):
            with self.assertRaises(UnsafeCodeError, msg=code):
                check_safety(code)

    def test_blocks_oversized_code(self):
        code = "\n".join(f"x{i} = {i}" for i in range(MAX_CODE_NODES))
        with self.assertRaises(UnsafeCodeError) as ctx:
            check_safety(code)
        self.assertIn("过于复杂", str(ctx.exception))

    def test_allows_common_pandas(self):
        for code in (
            'result = df.groupby("region")["amount"].sum().reset_index()',
            'result = df[df["amount"] > 1000].head(10)',
            'result = pd.to_datetime(df["order_date"]).dt.month.value_counts()',
            'result = df.pivot_table(index="region", columns="category", values="amount", aggfunc="sum")',
        ):
            check_safety(code)  # 不应抛异常


class TestExecuteGuards(unittest.TestCase):
    """执行层超时与截断。"""

    def setUp(self):
        self.df = pd.DataFrame({"amount": [1, 2, 3, 4, 5]})

    def test_normal_code_runs(self):
        res = execute_code('result = df["amount"].sum()', self.df)
        self.assertTrue(res["ok"])
        self.assertEqual(res["result"], 15)

    def test_long_loop_times_out(self):
        """死循环/长循环（模型最可能生成的卡死写法）必须能被超时打断。

        用纯 Python 循环而不是 `sum(range(N))`：后者是 C 层紧密循环、不释放 GIL，
        主线程要等它跑完才能恢复，属于线程方案的已知局限（强隔离见 Docker 方案）。
        """
        started = time.perf_counter()
        res = execute_code("result = 0\nfor i in range(20_000_000):\n    pass",
                           self.df, timeout=0.02)
        elapsed = time.perf_counter() - started
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage"], "timeout")
        self.assertLess(elapsed, 1.0, "超时后必须立刻返回")

    def test_result_truncated(self):
        big = pd.DataFrame({"x": range(100)})
        res = execute_code("result = df", big, max_rows=10)
        self.assertTrue(res["ok"])
        self.assertTrue(res["truncated"])
        self.assertEqual(len(res["result"]), 10)

    def test_unsafe_code_not_executed(self):
        res = execute_code('result = open("x.txt")', self.df)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage"], "safety")


class TestSqlTimeout(unittest.TestCase):
    """SQL 超时：长递归/大笛卡尔积应被中断，而不是一直占着 CPU。"""

    def test_long_query_interrupted(self):
        sql = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL "
               "SELECT x + 1 FROM c WHERE x < 100000000) SELECT count(*) FROM c")
        started = time.perf_counter()
        res = run_sql(sql, get_db_path(), timeout=0.2)
        elapsed = time.perf_counter() - started
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage"], "timeout")
        self.assertLess(elapsed, 5.0, "超时中断应及时生效")

    def test_normal_query_unaffected(self):
        res = run_sql("SELECT COUNT(*) FROM orders", get_db_path(), timeout=5)
        self.assertTrue(res["ok"])
        self.assertEqual(int(res["df"].iloc[0, 0]), 3877)


class TestLogging(unittest.TestCase):
    """运行日志。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agent-log-")
        import app.logging_utils as lu
        self.lu = importlib.reload(lu)
        self.lu.LOG_DIR = self.tmp
        self.lu.reset_logger()

    def tearDown(self):
        self.lu.reset_logger()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_jsonl(self):
        self.lu.log_run(ok=True, route="sql", retries=0, elapsed=1.23,
                        question="各区域销售额", rows=5)
        path = os.path.join(self.tmp, "agent.jsonl")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        self.assertTrue(record["ok"])
        self.assertEqual(record["route"], "sql")
        self.assertIn("ts", record)

    def test_broken_dir_does_not_raise(self):
        # 指向一个不可创建的路径，日志必须静默失败，不能影响问答主流程
        self.lu.LOG_DIR = os.path.join(self.tmp, "nope", "\0bad")
        self.lu.reset_logger()
        self.lu.log_run(ok=False)   # 不应抛异常


if __name__ == "__main__":
    unittest.main()
