"""链路追踪的离线测试（P2-2）：全部 mock 掉 LLM，不调用模型、不花钱。

覆盖：
- 追踪号生成（格式、同一秒内不撞号）
- 阶段事件落盘字段，且不会被 `read_runs()` 当成"一次问答"混进 logs 表格
- `read_events()` 精确匹配与前缀回退、顺序即执行顺序
- 状态机里四个阶段同挂一个 trace_id，汇总行同号
- 线性编排（`--linear`）同样有链，SQL 失败回退 pandas 时两组 generate/execute 都在
- 重试环、澄清分支、节点抛异常三种非顺利路径都要留痕
- `cli logs --trace` 的链路视图与找不到时的退出码
"""
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

try:
    import _env  # noqa: F401  测试环境隔离（临时 AGENT_HOME、关闭 SQL 复用）
except ModuleNotFoundError:      # 以包方式运行（python -m unittest tests.test_xxx）
    from . import _env  # noqa: F401
except ModuleNotFoundError:      # 以包方式运行（python -m unittest tests.test_xxx）
    from . import _env  # noqa: F401

import app.agent as A
import app.graph as G
import app.logging_utils as lu
import pandas as pd
from app.cli import EXIT_OK, EXIT_USAGE, _show_trace

GOOD_SQL = ("SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额 "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.region ORDER BY 销售额 DESC")
GOOD_PANDAS = 'result = df.groupby("region")["amount"].sum().reset_index()'
BAD_SQL = "SELECT nope FROM orders"
BAD_CODE = 'result = df["nope"].sum()'

TRACE_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


class TraceLogCase(unittest.TestCase):
    """把日志目录指到临时目录，避免测试污染开发机的 output/logs。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="trace-log-")
        self._old_dir = lu.LOG_DIR
        lu.LOG_DIR = self.tmp
        lu.reset_logger()

    def tearDown(self):
        lu.reset_logger()
        lu.LOG_DIR = self._old_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    @property
    def log_file(self) -> str:
        return os.path.join(self.tmp, "agent.jsonl")

    def records(self) -> list:
        if not os.path.exists(self.log_file):
            return []
        with open(self.log_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class TestTraceId(unittest.TestCase):
    def test_format(self):
        trace = lu.new_trace_id()
        self.assertRegex(trace, TRACE_RE)

    def test_unique_within_same_second(self):
        """同一秒内连问两次也不能撞号 —— 撞号会把两条链混成一条。"""
        ids = {lu.new_trace_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)


class TestLogEvent(TraceLogCase):
    def test_writes_stage_record(self):
        lu.log_event("20261124-101010-abcdef", "generate", ms=12, statement_len=88)
        record = self.records()[0]
        self.assertEqual(record["trace_id"], "20261124-101010-abcdef")
        self.assertEqual(record["stage"], "generate")
        self.assertEqual(record["ms"], 12)
        self.assertEqual(record["statement_len"], 88)
        self.assertIn("ts", record)

    def test_broken_dir_does_not_raise(self):
        lu.LOG_DIR = os.path.join(self.tmp, "nope", "\0bad")
        lu.reset_logger()
        lu.log_event("x", "generate", ms=1)   # 不应抛异常

    def test_events_are_not_counted_as_runs(self):
        """回归：阶段事件混进 logs 表格会让人误以为"每次问答都失败"多次。"""
        lu.log_event("t1", "understand", ms=1)
        lu.log_run(ok=True, trace_id="t1", route="sql")
        lu.log_event("t1", "answer", ms=1)

        runs = lu.read_runs(limit=0, path=self.log_file)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["trace_id"], "t1")


class TestReadEvents(TraceLogCase):
    def setUp(self):
        super().setUp()
        lu.log_event("20261124-101010-aaaaaa", "understand", ms=3)
        lu.log_event("20261124-101010-aaaaaa", "generate", ms=5)
        lu.log_event("20261124-101010-bbbbbb", "understand", ms=1)
        lu.log_run(ok=True, trace_id="20261124-101010-aaaaaa")

    def test_exact_match_keeps_order(self):
        events = lu.read_events("20261124-101010-aaaaaa", path=self.log_file)
        self.assertEqual([e["stage"] for e in events], ["understand", "generate"])

    def test_prefix_match_when_truncated(self):
        """表格里的追踪号可能被截断，复制前半段也要查得到。"""
        events = lu.read_events("20261124-101010-a", path=self.log_file)
        self.assertEqual({e["trace_id"] for e in events}, {"20261124-101010-aaaaaa"})

    def test_suffix_match_when_only_tail_remembered(self):
        """只记得尾巴上几位（`240001` 这种）也要查得到，查不到比查重更糟。"""
        events = lu.read_events("aaaaaa", path=self.log_file)
        self.assertEqual({e["trace_id"] for e in events}, {"20261124-101010-aaaaaa"})

    def test_exact_match_wins_over_prefix(self):
        # 加一条"是别人前缀"的记录，精确匹配必须优先
        lu.log_event("20261124-101010-aaaaaa-x", "answer", ms=1)
        events = lu.read_events("20261124-101010-aaaaaa", path=self.log_file)
        self.assertEqual(len(events), 2)

    def test_unknown_trace_returns_empty(self):
        self.assertEqual(lu.read_events("不存在", path=self.log_file), [])

    def test_substring_may_hit_multiple_chains(self):
        """两条链共有的那一段（时间部分）会同时命中 —— 交给展示层提醒，不静默合并。"""
        events = lu.read_events("101010", path=self.log_file)
        self.assertEqual(len({e["trace_id"] for e in events}), 2)

    def test_missing_file(self):
        self.assertEqual(lu.read_events("x", path=os.path.join(self.tmp, "无.jsonl")), [])


class TestGraphTrace(TraceLogCase):
    """跑真实状态机（LLM 全 mock），看链是不是一条。"""

    def _run(self, question="各区域的销售额是多少？", sql=GOOD_SQL, code=None,
             max_retries=1):
        agent = G.GraphAgent(with_chart=False, with_insight=False,
                             max_retries=max_retries)
        sql_mock = mock.Mock(return_value=sql)
        with mock.patch.object(G, "generate_sql", sql_mock):
            with mock.patch.object(G, "generate_code", mock.Mock(return_value=code or BAD_CODE)):
                return agent.run(question)

    def stages_of(self, trace_id: str) -> list:
        return [e["stage"] for e in lu.read_events(trace_id, path=self.log_file)]

    def test_happy_path_has_full_chain(self):
        state = self._run()
        trace = state["trace_id"]
        # cache 是 0.15.0 新增的第一段：先看能不能复用历史 SQL，没有再理解问题。
        # 即使没命中它也留一行 —— "为什么这次没复用"就在这一行里。
        self.assertEqual(self.stages_of(trace),
                         ["cache", "understand", "generate", "execute", "answer"])

        # 汇总行必须是同一个追踪号，否则"结果"和"过程"对不上号
        summaries = lu.read_runs(limit=0, path=self.log_file)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["trace_id"], trace)

    def test_events_record_useful_fields(self):
        state = self._run()
        events = {e["stage"]: e for e in lu.read_events(state["trace_id"], path=self.log_file)}
        self.assertEqual(events["understand"]["route"], "sql")
        self.assertGreater(events["generate"]["statement_len"], 0)
        self.assertEqual(events["execute"]["rows"], 5)
        # 缓存没命中时要留下原因（测试环境默认关闭复用，见 tests/_env.py）
        self.assertTrue(events["cache"].get("cache_note"))
        # 语句原文绝不进日志
        self.assertNotIn("SELECT", json.dumps(events, ensure_ascii=False))

    def test_retry_loop_leaves_reflect_in_chain(self):
        state = self._run(sql=BAD_SQL, code=BAD_CODE, max_retries=1)
        stages = self.stages_of(state["trace_id"])
        self.assertIn("reflect", stages)
        self.assertGreaterEqual(stages.count("generate"), 2)
        self.assertEqual(stages[-1], "answer")

    def test_clarify_chain_skips_generate(self):
        """取值对不上时只走 understand → clarify，不该出现"生成过"的痕迹。"""
        state = self._run(question="办公用品类产品的销售额是多少？")
        stages = self.stages_of(state["trace_id"])
        self.assertEqual(stages, ["cache", "understand", "clarify"])

    def test_two_runs_get_different_traces(self):
        first = self._run()["trace_id"]
        second = self._run()["trace_id"]
        self.assertNotEqual(first, second)
        self.assertEqual(len(lu.read_events(first, path=self.log_file)), 5)
        self.assertEqual(len(lu.read_events(second, path=self.log_file)), 5)

    def test_node_exception_is_recorded_and_raised(self):
        """节点抛异常时链里留一条 failed 事件，否则这条链会缺一段、看不出断在哪。"""
        def boom(state):
            raise RuntimeError("炸了")

        with mock.patch.object(G, "make_understand_node", return_value=lambda s: boom(s)):
            with self.assertRaises(RuntimeError):
                self._run()

        events = lu.read_events("", path=self.log_file)
        self.assertTrue(events)
        self.assertEqual(events[-1]["stage"], "understand")
        self.assertTrue(events[-1]["failed"])
        self.assertIn("RuntimeError", events[-1]["error"])


class TestTraceFields(unittest.TestCase):
    """字段白名单（P2-2）：两套引擎共用一份，敏感内容在源头就挡住。"""

    def test_statement_only_keeps_length(self):
        fields = lu.trace_fields({"statement": "SELECT * FROM orders WHERE amount > 1"})
        self.assertEqual(fields, {"statement_len": 37})
        self.assertNotIn("SELECT", json.dumps(fields, ensure_ascii=False))

    def test_result_only_keeps_rows(self):
        fields = lu.trace_fields({"result": pd.DataFrame({"a": [1, 2, 3]})})
        self.assertEqual(fields, {"rows": 3})

    def test_empty_result_still_reports_zero_rows(self):
        """rows=0 是结论（查不到数据），不能像其他零值一样被吞掉。"""
        self.assertEqual(lu.trace_fields({"result": pd.DataFrame()}), {"rows": 0})

    def test_zero_and_false_are_dropped(self):
        fields = lu.trace_fields({"retry_count": 0, "defer": False, "error": "",
                                  "chart_type": "", "route": "sql"})
        self.assertEqual(fields, {"route": "sql"})

    def test_unknown_keys_are_dropped(self):
        """问题原文、结果表这类不在白名单里的键，一律不落盘。"""
        fields = lu.trace_fields({"question": "各区域销售额", "secret": "x"}, route="sql")
        self.assertEqual(fields, {"route": "sql"})

    def test_extra_flag_passes_only_when_true(self):
        self.assertEqual(lu.trace_fields({"route": "pandas"}, fallback=True),
                         {"route": "pandas", "fallback": True})
        self.assertEqual(lu.trace_fields({"route": "pandas"}, fallback=False),
                         {"route": "pandas"})

    def test_bad_value_is_skipped_not_raised(self):
        """类型不对只跳过这个字段，不能让整条日志丢掉。"""
        self.assertEqual(lu.trace_fields({"lessons_used": 42}), {})

    def test_both_engines_use_the_same_whitelist(self):
        """回归：状态机与线性编排不许各写一份字段映射。"""
        self.assertFalse(hasattr(G, "_TRACE_FIELDS"))
        self.assertFalse(hasattr(G, "_trace_fields"))
        self.assertTrue(callable(lu.trace_fields))


class TestLinearTrace(TraceLogCase):
    """线性编排（`--linear`）也要有链 —— 否则两种引擎的排查方式不一致。"""

    def _run(self, question="各区域的销售额是多少？", sql=GOOD_SQL,
             code=None, force_pandas=False):
        agent = A.DataAgent(with_chart=False, with_insight=False,
                            force_pandas=force_pandas)
        with mock.patch.object(A, "generate_sql", mock.Mock(return_value=sql)), \
                mock.patch.object(A, "generate_code",
                                  mock.Mock(return_value=code or BAD_CODE)):
            return agent.run(question, [])

    def stages_of(self, trace_id: str) -> list:
        return [e["stage"] for e in lu.read_events(trace_id, path=self.log_file)]

    def test_happy_path_chain(self):
        state = self._run()
        trace = state["trace_id"]
        self.assertEqual(self.stages_of(trace), ["generate", "execute", "answer"])

        summaries = lu.read_runs(limit=0, path=self.log_file)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["trace_id"], trace)
        self.assertEqual(summaries[0]["engine"], "linear")
        self.assertEqual(summaries[0]["rows"], 5)

    def test_statement_text_not_logged(self):
        state = self._run()
        events = lu.read_events(state["trace_id"], path=self.log_file)
        self.assertNotIn("SELECT", json.dumps(events, ensure_ascii=False))
        self.assertGreater(events[0]["statement_len"], 0)

    def test_summary_does_not_fake_token_usage(self):
        """线性版没挂 token 回调：不写 llm_calls/tokens，别用 0/0 假装统计过。"""
        self._run()
        summary = lu.read_runs(limit=0, path=self.log_file)[0]
        self.assertNotIn("llm_calls", summary)
        self.assertNotIn("tokens_in", summary)

    def test_sql_failure_falls_back_to_pandas(self):
        """链里两组 generate/execute 就是"回退过"，第二组带 fallback 标记。"""
        state = self._run(sql=BAD_SQL, code=GOOD_PANDAS)
        self.assertEqual(state["route"], "pandas")
        self.assertEqual(self.stages_of(state["trace_id"]),
                         ["generate", "execute", "generate", "execute", "answer"])

        events = lu.read_events(state["trace_id"], path=self.log_file)
        self.assertTrue(events[1].get("failed"))
        self.assertTrue(events[3].get("fallback"))

    def test_total_failure_ends_with_failed_answer(self):
        """两条路都没成时链尾要留一行失败，否则链看起来"没走完"。"""
        state = self._run(sql=BAD_SQL, code=BAD_CODE)
        self.assertIsNone(state["result"])
        events = lu.read_events(state["trace_id"], path=self.log_file)
        self.assertEqual(events[-1]["stage"], "answer")
        self.assertTrue(events[-1]["failed"])

        summary = lu.read_runs(limit=0, path=self.log_file)[0]
        self.assertFalse(summary["ok"])
        self.assertIn("失败", summary["error"])

    def test_force_pandas_skips_sql_attempt(self):
        state = self._run(force_pandas=True, code=GOOD_PANDAS)
        self.assertEqual(state["route"], "pandas")
        self.assertEqual(self.stages_of(state["trace_id"]),
                         ["generate", "execute", "answer"])

    def test_run_returns_state_for_chat(self):
        """回归：`chat --linear` 调的是 agent.run()，DataAgent 原先只有 ask()。"""
        state = self._run()
        for key in ("route", "result", "trace_id", "suggestions", "answer"):
            self.assertIn(key, state)
        self.assertIsInstance(state["suggestions"], list)
        self.assertTrue(state["trace_id"])


class TestShowTrace(TraceLogCase):
    """`cli logs --trace` 的输出。"""

    def setUp(self):
        super().setUp()
        lu.log_event("20261124-101010-aaaaaa", "understand", ms=3, route="sql")
        lu.log_event("20261124-101010-aaaaaa", "execute", ms=7, rows=5)
        lu.log_run(ok=True, trace_id="20261124-101010-aaaaaa", rows=5,
                   llm_calls=2, tokens_in=100, tokens_out=20, elapsed=0.12)

    def _call(self, trace_id):
        # cli 里是 from-import，所以要替换它命名空间里的引用
        with mock.patch("app.cli.read_events",
                        side_effect=lambda t: lu.read_events(t, path=self.log_file)), \
                mock.patch("app.cli.read_runs",
                           side_effect=lambda **kw: lu.read_runs(limit=0, path=self.log_file)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                code = _show_trace(trace_id)
        return code, buf.getvalue()

    def test_prints_stage_chain(self):
        code, out = self._call("20261124-101010-aaaaaa")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("understand", out)
        self.assertIn("route=sql", out)
        self.assertIn("rows=5", out)
        self.assertIn("阶段合计 10ms", out)
        self.assertIn("成功", out)          # 末尾带上汇总行

    def test_missing_trace_returns_usage_error(self):
        code, _ = self._call("不存在的号")
        self.assertEqual(code, EXIT_USAGE)

    def test_linear_summary_says_token_not_counted(self):
        """线性汇总行没有 token 字段：显示"未统计"，不能显示 0/0。"""
        lu.log_event("20261124-101010-cccccc", "generate", ms=5, route="sql")
        lu.log_run(ok=True, trace_id="20261124-101010-cccccc", rows=3,
                   elapsed=0.2, engine="linear")
        code, out = self._call("20261124-101010-cccccc")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("token 未统计（线性路线）", out)
        self.assertIn("（线性编排）", out)


if __name__ == "__main__":
    unittest.main()
