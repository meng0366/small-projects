"""命令行与阶段 6 新模块的离线测试。

覆盖：
- 参数解析与旧命令兼容（`python -m app.cli "问题"` 仍等价 `ask`）
- 输出渲染与导出（markdown 转义、JSON 的 NaN、CSV BOM、xlsx）
- 会话持久化（保存 / 续聊 / 列表 / 坏文件容错 / 非法会话名）
- 运行日志读取（坏行跳过、--failed、-n）
- 追问建议（不推荐已经看过的维度、不推荐"只看当前对象"）
- 命令行端到端（mock 掉状态机，不调模型）

全部离线，不需要 API Key。
"""
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd

from app import cli, sessions
from app.logging_utils import read_runs
from app.render import (
    UnsupportedFormat,
    export,
    normalize_format,
    render,
    to_json,
    to_markdown,
    to_table,
)
from app.tools.suggest_tool import suggest_followups


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "区域": ["华南", "西南", "华中"],
        "销售额": [5064245.4, 4986839.0, None],
    })


class TestArgv(unittest.TestCase):
    """旧写法必须还能用，否则 README 里的命令全失效。"""

    def test_no_args_goes_to_chat(self):
        self.assertEqual(cli.normalize_argv([]), ["chat"])

    def test_bare_question_becomes_ask(self):
        self.assertEqual(cli.normalize_argv(["各区域销售额"]), ["ask", "各区域销售额"])

    def test_flags_before_question(self):
        self.assertEqual(cli.normalize_argv(["--no-chart", "算一下"]),
                         ["ask", "--no-chart", "算一下"])

    def test_known_subcommand_kept(self):
        for name in ("ask", "chat", "schema", "logs", "version", "c", "l"):
            self.assertEqual(cli.normalize_argv([name]), [name])

    def test_help_kept(self):
        self.assertEqual(cli.normalize_argv(["-h"]), ["-h"])


class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = cli.build_parser()

    def test_ask_options(self):
        args = self.parser.parse_args(["ask", "问题", "-f", "json", "-o", "a.csv",
                                       "--db", "x.db", "--no-chart", "-q"])
        self.assertEqual(args.command, "ask")
        self.assertEqual(args.question, "问题")
        self.assertEqual(args.format, "json")
        self.assertEqual(args.out, "a.csv")
        self.assertEqual(args.db, "x.db")
        self.assertTrue(args.no_chart)
        self.assertTrue(args.quiet)

    def test_ask_alias(self):
        args = self.parser.parse_args(["a", "问题"])
        self.assertEqual(args.question, "问题")

    def test_chat_options(self):
        args = self.parser.parse_args(["chat", "-c", "--session", "测试"])
        self.assertTrue(args.resume)
        self.assertEqual(args.session, "测试")

    def test_logs_options(self):
        args = self.parser.parse_args(["logs", "-n", "5", "--failed", "--json"])
        self.assertEqual(args.n, 5)
        self.assertTrue(args.failed)
        self.assertTrue(args.json)
        self.assertEqual(args.trace, "")

    def test_logs_trace_option(self):
        """`-t` 与 `--trace` 都要能指定追踪号（P2-2）。"""
        for flag in ("-t", "--trace"):
            args = self.parser.parse_args(["logs", flag, "20261124-101010-abcd"])
            self.assertEqual(args.trace, "20261124-101010-abcd")


class TestRender(unittest.TestCase):
    def test_normalize_format_aliases(self):
        self.assertEqual(normalize_format(None), "table")
        self.assertEqual(normalize_format("MD"), "markdown")
        self.assertEqual(normalize_format("json"), "json")

    def test_normalize_format_rejects_unknown(self):
        with self.assertRaises(UnsupportedFormat):
            normalize_format("yaml")

    def test_table_truncates(self):
        text = to_table(sample_df(), max_rows=2)
        self.assertIn("仅显示前 2 行", text)

    def test_markdown_escapes_pipe(self):
        df = pd.DataFrame({"列|名": ["a|b"]})
        text = to_markdown(df)
        self.assertIn("列\\|名", text)
        self.assertIn("| --- |", text)

    def test_json_nulls_nan(self):
        data = json.loads(to_json(sample_df()))
        self.assertIsNone(data[2]["销售额"])
        self.assertEqual(data[0]["区域"], "华南")

    def test_csv_has_no_index(self):
        self.assertTrue(render(sample_df(), "csv").startswith("区域,销售额"))

    def test_render_unknown_format(self):
        with self.assertRaises(UnsupportedFormat):
            render(sample_df(), "xlsx-please")


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_csv_has_bom_for_excel(self):
        path = export(sample_df(), str(self.root / "out.csv"))
        self.assertTrue(Path(path).exists())
        with open(path, "rb") as f:
            self.assertTrue(f.read(3) == b"\xef\xbb\xbf", "CSV 需要 BOM，Excel 才不乱码")

    def test_export_json(self):
        path = export(sample_df(), str(self.root / "out.json"))
        with open(path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), 3)

    def test_export_markdown(self):
        path = export(sample_df(), str(self.root / "out.md"))
        self.assertIn("| 区域 |", Path(path).read_text(encoding="utf-8"))

    def test_export_xlsx(self):
        path = export(sample_df(), str(self.root / "out.xlsx"))
        back = pd.read_excel(path)
        self.assertEqual(list(back.columns), ["区域", "销售额"])
        self.assertEqual(len(back), 3)

    def test_export_creates_missing_dir(self):
        path = export(sample_df(), str(self.root / "深" / "层" / "out.csv"))
        self.assertTrue(Path(path).exists())

    def test_export_unknown_suffix(self):
        with self.assertRaises(UnsupportedFormat):
            export(sample_df(), str(self.root / "out.txt"))


class TestSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"AGENT_HOME": self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_save_and_load_roundtrip(self):
        sessions.save_session("演示", [{"question": "q1", "answer": "a1"}])
        loaded = sessions.load_session("演示")
        self.assertEqual(loaded["turns"][0]["answer"], "a1")
        self.assertTrue(loaded["created"])

    def test_append_turn_creates_then_extends(self):
        sessions.append_turn("s1", "第一问", "第一答")
        sessions.append_turn("s1", "第二问", "第二答")
        self.assertEqual(len(sessions.load_session("s1")["turns"]), 2)

    def test_load_latest_without_name(self):
        sessions.save_session("a", [])
        sessions.save_session("b", [])
        # 显式拉开更新时间，避免两次写盘落在同一时间精度里导致结果不稳定
        old = os.path.join(sessions.sessions_dir(), "a.json")
        os.utime(old, (1_700_000_000, 1_700_000_000))
        self.assertEqual(sessions.load_session(None)["name"], "b")

    def test_list_sessions_summary(self):
        sessions.append_turn("x", "最后的问题", "答")
        rows = sessions.list_sessions()
        self.assertEqual(rows[0]["name"], "x")
        self.assertEqual(rows[0]["turns"], 1)
        self.assertEqual(rows[0]["last"], "最后的问题")

    def test_reject_bad_name(self):
        for bad in ("../逃逸", "a/b", "", "x" * 60):
            with self.assertRaises(sessions.SessionError):
                sessions.save_session(bad, [])

    def test_corrupted_file_is_tolerated_by_list(self):
        directory = sessions.sessions_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "broken.json").write_text("{不是 JSON", encoding="utf-8")
        rows = sessions.list_sessions()
        self.assertEqual(rows[0]["name"], "broken")
        with self.assertRaises(sessions.SessionError):
            sessions.load_session("broken")

    def test_load_missing_session(self):
        with self.assertRaises(sessions.SessionError):
            sessions.load_session("不存在")

    def test_turns_to_history_keeps_last_turns(self):
        turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)]
        history = sessions.turns_to_history(turns, max_turns=2)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "q4")
        self.assertEqual(history[-1]["role"], "assistant")


class TestReadRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "agent.jsonl")
        lines = [
            json.dumps({"ts": "t1", "ok": True, "route": "sql", "elapsed": 1.0}),
            "这不是 JSON",
            json.dumps({"ts": "t2", "ok": False, "route": "pandas", "error": "boom"}),
            json.dumps({"ts": "t3", "ok": True, "route": "sql"}),
        ]
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_skips_bad_lines(self):
        rows = read_runs(path=self.path)
        self.assertEqual(len(rows), 3)

    def test_limit_returns_tail(self):
        rows = read_runs(limit=1, path=self.path)
        self.assertEqual(rows[0]["ts"], "t3")

    def test_failed_only(self):
        rows = read_runs(path=self.path, failed_only=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ts"], "t2")

    def test_missing_file(self):
        self.assertEqual(read_runs(path=os.path.join(self.tmp.name, "无.jsonl")), [])


class TestSuggest(unittest.TestCase):
    def test_suggests_other_dimensions(self):
        items = suggest_followups("各区域的销售额是多少？", sample_df())
        self.assertLessEqual(len(items), 3)
        self.assertFalse(any("各区域" in item for item in items), "已经看区域了不该再推荐区域")
        self.assertTrue(any("品类" in item for item in items))

    def test_narrow_to_first_value(self):
        items = suggest_followups("销售额怎么样？", sample_df())
        self.assertTrue(any("华南" in item for item in items))

    def test_no_narrow_when_already_scoped(self):
        df = pd.DataFrame({"区域": ["华东"], "销售额": [993052.95]})
        items = suggest_followups("只看华东的销售额是多少？", df)
        self.assertFalse(any("只看「华东」" in item for item in items))

    def test_time_suggestion_only_once(self):
        items = suggest_followups("每个月的销售额走势如何？", sample_df())
        self.assertFalse(any("走势" in item for item in items))

    def test_handles_none_result(self):
        items = suggest_followups("算一下", None)
        self.assertTrue(items)
        self.assertLessEqual(len(items), 3)

    def test_respects_max_suggestions(self):
        self.assertEqual(len(suggest_followups("问题", sample_df(), max_suggestions=1)), 1)
        self.assertEqual(suggest_followups("问题", sample_df(), max_suggestions=0), [])


def fake_state(df=None) -> dict:
    return {
        "question": "各区域的销售额是多少？",
        "route": "sql",
        "statement": "SELECT 1",
        "result": df,
        "insight": "华南最高",
        "answer": "" if df is not None else "未能完成分析。",
        "errors": [] if df is not None else ["SQL 执行失败"],
        "retry_count": 0,
        "suggestions": ["各品类的销售额是多少？"],
        "_elapsed": 0.9,
    }


class TestCliEndToEnd(unittest.TestCase):
    """mock 掉 GraphAgent，验证参数→输出→退出码这条链路。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"AGENT_HOME": str(self.root / "home")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _run(self, argv, state):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli, "GraphAgent") as agent_cls:
            agent_cls.return_value.run.return_value = state
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_json_output(self):
        code, out, _ = self._run(["ask", "各区域的销售额是多少？", "-f", "json"],
                                 fake_state(sample_df()))
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["rows"], 3)
        self.assertEqual(payload["route"], "sql")
        self.assertEqual(payload["suggestions"], ["各品类的销售额是多少？"])
        self.assertEqual(payload["data"][0]["区域"], "华南")

    def test_attribution_outputs_table_and_json(self):
        """归因结果要在两处露出来：人看的加减表，脚本看的 attribution 字段。"""
        state = fake_state(sample_df())
        state["attribution"] = {
            "ok": True, "dim": "品类", "metric": "销售额",
            "base_label": "5月", "current_label": "6月",
            "base_total": 100.0, "current_total": 120.0, "delta": 20.0, "pct": 0.2,
            "additive": True, "notes": [],
            "table": pd.DataFrame({"维度": ["手机"], "基期": [100.0], "本期": [120.0],
                                   "变化": [20.0], "贡献%": [100.0]}),
        }
        code, out, _ = self._run(["ask", "q"], state)
        self.assertEqual(code, 0)
        self.assertIn("[归因]", out)

        code, out, _ = self._run(["ask", "q", "-f", "json"], state)
        payload = json.loads(out)
        self.assertEqual(payload["attribution"]["dimension"], "品类")
        self.assertEqual(payload["attribution"]["rows"][0]["变化"], 20.0)

    def test_table_output_contains_sections(self):
        code, out, _ = self._run(["ask", "各区域的销售额是多少？"], fake_state(sample_df()))
        self.assertEqual(code, 0)
        for marker in ("[路由]", "[生成的SQL]", "[分析结果]", "[洞察]", "[可以接着问]"):
            self.assertIn(marker, out)

    def test_csv_output_is_pure_data(self):
        code, out, _ = self._run(["ask", "q", "-f", "csv"], fake_state(sample_df()))
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("区域,销售额"), f"标准输出应只有数据：{out!r}")

    def test_export_writes_file_and_keeps_stdout_clean(self):
        target = self.root / "结果.csv"
        code, out, err = self._run(["ask", "q", "-f", "csv", "-o", str(target)],
                                   fake_state(sample_df()))
        self.assertEqual(code, 0)
        self.assertTrue(target.exists())
        self.assertIn("已导出", err)

    def test_failure_returns_exit_code_1(self):
        code, _, err = self._run(["ask", "q", "-f", "csv"], fake_state(None))
        self.assertEqual(code, 1)
        self.assertIn("未能完成分析", err)

    def test_missing_db_returns_exit_code_2(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cli.main(["ask", "q", "--db", str(self.root / "没有这个库.db")])
        self.assertEqual(code, 2)

    def test_empty_question_returns_usage_error(self):
        args = Namespace(question="", format="table", out=None, quiet=False,
                         db=None, no_chart=False, no_insight=False,
                         linear=False, pandas=False)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.run_ask(args), 2)

    def test_quiet_hides_statement(self):
        args = Namespace(question="q", format="table", out=None, quiet=True,
                         db=None, no_chart=False, no_insight=False,
                         linear=True, pandas=False)
        out = io.StringIO()
        with mock.patch.object(cli, "DataAgent") as agent_cls:
            # 线性引擎也走 run()，与状态机同一接口
            agent_cls.return_value.run.return_value = fake_state(sample_df())
            with redirect_stdout(out):
                code = cli.run_ask(args)
        self.assertEqual(code, 0)
        self.assertNotIn("SELECT 1", out.getvalue())
        self.assertIn("[分析结果]", out.getvalue())

    def test_logs_command_prints_rows(self):
        record = {"ts": "2026-11-24T18:00:00", "ok": True, "route": "sql",
                  "rows": 5, "elapsed": 1.1, "question": "各区域销售额"}
        out = io.StringIO()
        with mock.patch.object(cli, "read_runs", return_value=[record]), \
                mock.patch.object(cli, "log_path", return_value="C:/tmp/agent.jsonl"):
            with redirect_stdout(out):
                code = cli.run_logs(Namespace(n=5, failed=False, json=False))
        self.assertEqual(code, 0)
        self.assertIn("各区域销售额", out.getvalue())
        self.assertIn("共 1 条", out.getvalue())

    def test_schema_command_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.run_schema(Namespace(json=True, db=None))
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertIn("tables", payload)
        self.assertTrue(payload["mode"])


class TestAttributionOutput(unittest.TestCase):
    """归因在命令行 / JSON 里的展示（二级拆解要单独成段，不能糊进一级表）。"""

    def _att(self) -> dict:
        return {
            "ok": True, "dim": "品类", "dim2": "客户等级", "metric": "销售额",
            "base_label": "5月", "current_label": "6月",
            "base_total": 10.0, "current_total": 20.0, "delta": 10.0,
            "pct": 1.0, "additive": True, "notes": [],
            "table": pd.DataFrame({"维度": ["手机"], "基期": [10.0], "本期": [20.0],
                                   "变化": [10.0], "贡献%": [100.0]}),
            "drill": {
                "dim": "客户等级", "item": "手机", "item_delta": 10.0, "closure_ok": True,
                "table": pd.DataFrame({"维度": ["战略客户"], "基期": [5.0], "本期": [15.0],
                                       "变化": [10.0], "贡献%": [100.0]}),
            },
        }

    def test_payload_carries_drill(self):
        payload = cli._build_payload(
            {"attribution": self._att(), "result": pd.DataFrame({"维度": ["手机"]})}, "q")
        drill = payload["attribution"]["drill"]
        self.assertEqual(drill["dimension"], "客户等级")
        self.assertEqual(drill["item"], "手机")
        self.assertTrue(drill["closure_ok"])
        self.assertEqual(drill["rows"][0]["变化"], 10.0)

    def test_payload_without_drill_keeps_single_level(self):
        att = self._att()
        att.pop("drill")
        payload = cli._build_payload({"attribution": att,
                                      "result": pd.DataFrame({"维度": ["手机"]})}, "q")
        self.assertNotIn("drill", payload["attribution"])

    def test_human_output_prints_drill_block(self):
        out = io.StringIO()
        with redirect_stdout(out):
            cli._print_human({"attribution": self._att(),
                              "result": pd.DataFrame({"维度": ["手机"]})}, "q")
        text = out.getvalue()
        self.assertIn("[归因]", text)
        self.assertIn("[归因·下钻]", text)
        self.assertIn("战略客户", text)


if __name__ == "__main__":
    unittest.main()
