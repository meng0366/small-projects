"""MCP server 测试（离线，不调模型）。

重点覆盖三类容易出错、且出错后很难发现的地方：

1. **stdout 洁净**：stdio 传输下 stdout 是协议通道。任何一行 print 都会让客户端
   只看到"服务器启动失败"，而本地跑 `python -m app.cli serve` 完全正常 —— 这种
   bug 在人工测试里几乎看不出来，必须靠断言。
2. **错误必须结构化**：工具抛异常时，对面模型只看到 traceback，无法自我纠正；
   所以要求一律返回 `{"ok": false, "error": ...}`。
3. **截断要标注**：MCP 的返回值会进另一个模型的上下文，悄悄只给一部分数据比
   截断本身更危险。
"""
import asyncio
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pandas as pd
import yaml

from app import cli, payload as payload_mod
from app.mcp_server import (
    MAX_ROWS_DEFAULT,
    SERVER_NAME,
    ask_tool,
    build_server,
    catalog_tool,
    schema_tool,
    serve,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES_DB = os.path.join(REPO_ROOT, "data", "sales.db")


def _mcp_available() -> bool:
    """mcp 是可选依赖。没装时只跳过"真要建 server"的几条，其余照样跑。

    取舍：不把整个文件 skip 掉 —— 工具实现（参数映射、截断、错误结构化、stdout 洁净）
    都不依赖 mcp，没装也该被验证；只有"注册到协议层"这部分必须有 mcp。
    """
    try:
        from app.mcp_server import _server_class
        _server_class()
        return True
    except Exception:
        return False


HAS_MCP = _mcp_available()
NEEDS_MCP = "需要 mcp（pip install -r requirements-mcp.txt）"


def fake_state(rows=5, answer="", **extra):
    """造一份与 Agent 返回同形状的 state（字段名与 graph/agent 保持一致）。"""
    df = pd.DataFrame({
        "区域": [f"区{i}" for i in range(rows)],
        "销售额": [1000.0 + i for i in range(rows)],
    }) if rows else None
    state = {
        "result": df,
        "route": "sql",
        "statement": "SELECT 1",
        "answer": answer,
        "trace_id": "20261126-000000-000001",
        "plan": None,
        "plan_issues": [],
        "peek_used": False,
        "errors": [],
        "suggestions": [],
        "retry_count": 0,
        "_elapsed": 1.0,
    }
    state.update(extra)
    return state


@unittest.skipUnless(HAS_MCP, NEEDS_MCP)
class TestToolsRegistered(unittest.IsolatedAsyncioTestCase):
    def test_three_tools_with_descriptions(self):
        server = build_server()
        tools = asyncio.run(server.list_tools())
        names = [t.name for t in tools]
        self.assertEqual(names, ["ask", "get_schema", "generate_catalog"])
        for tool in tools:
            self.assertTrue((tool.description or "").strip(), tool.name)

    def test_ask_schema_exposes_expected_args(self):
        server = build_server()
        tools = {t.name: t for t in asyncio.run(server.list_tools())}
        # 1.x 叫 inputSchema，2.x 叫 input_schema
        schema = (getattr(tools["ask"], "input_schema", None)
                  or getattr(tools["ask"], "inputSchema", None))
        props = schema["properties"]
        for key in ("question", "db", "plan", "peek", "max_rows"):
            self.assertIn(key, props)
        self.assertEqual(schema["required"], ["question"])
        # 默认值要是"省 token"的那一档：洞察与图表都关
        self.assertFalse(props["insight"]["default"])
        self.assertFalse(props["chart"]["default"])
        self.assertTrue(props["plan"]["default"])

    def test_default_db_applies_to_every_tool(self):
        server = build_server(db=SALES_DB)
        tools = {t.name: t for t in asyncio.run(server.list_tools())}
        for name in ("ask", "get_schema"):
            schema = (getattr(tools[name], "input_schema", None)
                      or getattr(tools[name], "inputSchema", None))
            self.assertEqual(schema["properties"]["db"]["default"], SALES_DB)

    def test_instructions_mention_the_three_tools(self):
        server = build_server()
        text = getattr(server, "instructions", "") or ""
        for word in ("ask", "get_schema", "generate_catalog"):
            self.assertIn(word, text)


class TestAskTool(unittest.TestCase):
    def test_returns_payload_json(self):
        with patch("app.cli.ask_once", return_value=fake_state(rows=5)):
            data = json.loads(ask_tool("各区域销售额"))
        self.assertTrue(data["ok"])
        self.assertEqual(data["route"], "sql")
        self.assertEqual(data["rows"], 5)
        self.assertEqual(len(data["data"]), 5)
        self.assertEqual(data["columns"], ["区域", "销售额"])
        self.assertFalse(data["truncated"])
        self.assertEqual(data["trace_id"], "20261126-000000-000001")

    def test_chinese_is_not_escaped(self):
        # 返回值会被另一个模型直接阅读，\u 转义既费 token 又难读
        with patch("app.cli.ask_once", return_value=fake_state()):
            text = ask_tool("各区域销售额")
        self.assertIn("区域", text)
        self.assertNotIn("\\u", text)

    def test_empty_question(self):
        with patch("app.cli.ask_once", side_effect=AssertionError("不该调用模型")):
            data = json.loads(ask_tool("   "))
        self.assertFalse(data["ok"])
        self.assertIn("不能为空", data["error"])

    def test_missing_db_reports_instead_of_degrading(self):
        with patch("app.cli.ask_once", side_effect=AssertionError("不该调用模型")):
            data = json.loads(ask_tool("各区域销售额", db="不存在.db"))
        self.assertFalse(data["ok"])
        self.assertIn("数据库文件不存在", data["error"])

    def test_model_exception_becomes_readable_error(self):
        with patch("app.cli.ask_once", side_effect=RuntimeError("余额不足")):
            data = json.loads(ask_tool("各区域销售额"))
        self.assertFalse(data["ok"])
        self.assertIn("RuntimeError", data["error"])
        self.assertIn("余额不足", data["error"])

    def test_failed_run_surfaces_answer_as_error(self):
        state = fake_state(rows=0, answer="你提到的「京东」不在销售渠道取值中。")
        with patch("app.cli.ask_once", return_value=state):
            data = json.loads(ask_tool("只看京东渠道的销售额"))
        self.assertFalse(data["ok"])
        self.assertIn("京东", data["error"])

    def test_truncates_and_says_so(self):
        with patch("app.cli.ask_once", return_value=fake_state(rows=300)):
            data = json.loads(ask_tool("所有明细", max_rows=50))
        self.assertEqual(data["rows"], 300)          # 真实总数照实说
        self.assertEqual(len(data["data"]), 50)      # 只给 50 行
        self.assertTrue(data["truncated"])
        self.assertEqual(data["row_limit"], 50)

    def test_no_truncation_marker_when_under_limit(self):
        with patch("app.cli.ask_once", return_value=fake_state(rows=10)):
            data = json.loads(ask_tool("各区域销售额", max_rows=50))
        self.assertFalse(data["truncated"])
        self.assertEqual(len(data["data"]), 10)

    def test_switches_map_to_cli_flags(self):
        """工具参数最终要落成 CLI 认识的那几个开关，映射错了会静默失效。"""
        captured = {}

        def fake(args, question):
            captured.update(vars(args))
            return fake_state()

        with patch("app.cli.ask_once", side_effect=fake):
            ask_tool("q", plan=False, peek=False, insight=True, chart=True, linear=True)
        self.assertTrue(captured["no_plan"])
        self.assertTrue(captured["no_peek"])
        self.assertFalse(captured["no_insight"])     # insight=True → 不传 --no-insight
        self.assertFalse(captured["no_chart"])
        self.assertTrue(captured["linear"])

    def test_defaults_keep_cheap_path(self):
        captured = {}

        def fake(args, question):
            captured.update(vars(args))
            return fake_state()

        with patch("app.cli.ask_once", side_effect=fake):
            ask_tool("q")
        self.assertFalse(captured["no_plan"])        # 默认开计划
        self.assertFalse(captured["no_peek"])
        self.assertTrue(captured["no_insight"])      # 默认不出洞察
        self.assertTrue(captured["no_chart"])        # 默认不画图


class TestSchemaTool(unittest.TestCase):
    def test_real_db(self):
        data = json.loads(schema_tool(db=SALES_DB))
        self.assertTrue(data["ok"])
        names = [t["name"] for t in data["tables"]]
        self.assertIn("orders", names)
        orders = next(t for t in data["tables"] if t["name"] == "orders")
        cols = {c["name"]: c for c in orders["columns"]}
        self.assertIn("amount", cols)
        self.assertTrue(cols["amount"]["label"])     # 中文名来自 catalog
        self.assertIn("销售额", data["terms"])

    def test_default_source(self):
        data = json.loads(schema_tool())
        self.assertTrue(data["ok"])
        self.assertTrue(data["tables"])

    def test_missing_db(self):
        data = json.loads(schema_tool(db="没有这个库.db"))
        self.assertFalse(data["ok"])
        self.assertIn("数据库文件不存在", data["error"])


class TestCatalogTool(unittest.TestCase):
    def test_dry_run_never_calls_model(self):
        with patch("app.mcp_server.generate_descriptions",
                   side_effect=AssertionError("dry-run 不该调模型")):
            data = json.loads(catalog_tool(db=SALES_DB, dry_run=True))
        self.assertTrue(data["ok"])
        self.assertTrue(data["dry_run"])
        self.assertTrue(data["tables"])
        self.assertIn("张表", data["summary"])

    def test_writes_draft_next_to_db(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "demo.db")
            import shutil
            shutil.copy(SALES_DB, db)
            raw = json.dumps({"tables": [
                {"name": "orders", "label": "订单表", "description": "一行一笔订单",
                 "columns": {"amount": {"label": "销售额", "note": "已乘折扣"}}},
            ], "terms": [{"name": "销售额", "definition": "SUM(orders.amount)",
                          "source": "orders.amount"}]}, ensure_ascii=False)
            with patch("app.mcp_server.generate_descriptions", return_value=raw):
                data = json.loads(catalog_tool(db=db))

            self.assertTrue(data["ok"])
            self.assertEqual(data["path"], os.path.join(tmp, "catalog.draft.yaml"))
            self.assertTrue(os.path.exists(data["path"]))
            # 产物必须能被现有 loader 直接消费
            with open(data["path"], encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            self.assertEqual([t["name"] for t in loaded["tables"]],
                             ["customers", "orders", "products"])
            orders = next(t for t in loaded["tables"] if t["name"] == "orders")
            self.assertEqual(orders["columns"]["amount"]["label"], "销售额")
            self.assertIn("销售额", loaded["terms"])

    def test_missing_db(self):
        data = json.loads(catalog_tool(db=""))
        self.assertFalse(data["ok"])
        data = json.loads(catalog_tool(db="没有这个库.db"))
        self.assertFalse(data["ok"])

    def test_unparsable_model_output_writes_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "x.yaml")
            with patch("app.mcp_server.generate_descriptions", return_value="我不会写 JSON"):
                data = json.loads(catalog_tool(db=SALES_DB, out=out))
            self.assertFalse(data["ok"])
            self.assertFalse(os.path.exists(out))


class TestStdoutStaysClean(unittest.TestCase):
    """stdio 传输下 stdout 是 JSON-RPC 通道，这里一旦有输出客户端就连不上。"""

    def _assert_silent(self, fn, *a, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*a, **kw)
        self.assertEqual(buf.getvalue(), "", f"{fn.__name__} 往 stdout 写了东西")

    def test_ask_is_silent(self):
        with patch("app.cli.ask_once", return_value=fake_state()):
            self._assert_silent(ask_tool, "各区域销售额")

    def test_ask_silent_on_failure(self):
        with patch("app.cli.ask_once", side_effect=RuntimeError("boom")):
            self._assert_silent(ask_tool, "各区域销售额")

    def test_schema_is_silent(self):
        self._assert_silent(schema_tool, db=SALES_DB)

    def test_catalog_dry_run_is_silent(self):
        self._assert_silent(catalog_tool, db=SALES_DB, dry_run=True)


class TestServeGuards(unittest.TestCase):
    def test_missing_mcp_message_points_to_requirements(self):
        """mcp 是可选依赖：没装时要给出可执行的下一步，而不是一句 ModuleNotFoundError。"""
        import sys
        from app.mcp_server import _server_class
        hidden = {k: None for k in
                  ("mcp", "mcp.server", "mcp.server.mcpserver", "mcp.server.fastmcp")}
        with patch.dict(sys.modules, hidden):
            with self.assertRaises(ModuleNotFoundError) as cm:
                _server_class()
        self.assertIn("requirements-mcp", str(cm.exception))

    def test_serve_reports_missing_mcp_on_stderr(self):
        """没装 mcp 时只提示、不抛栈，也不往 stdout 写（客户端可能已在等握手）。"""
        err, out = io.StringIO(), io.StringIO()
        with patch("app.mcp_server._server_class",
                   side_effect=ModuleNotFoundError(
                       "没有安装 mcp。请 pip install -r requirements-mcp.txt")):
            with redirect_stderr(err), redirect_stdout(out):
                code = serve()
        self.assertEqual(code, 1)
        self.assertIn("启动失败", err.getvalue())
        self.assertIn("requirements-mcp", err.getvalue())
        self.assertEqual(out.getvalue(), "")

    def test_cli_parser_has_serve(self):
        args = cli.build_parser().parse_args(["serve"])
        self.assertEqual(args.command, "serve")
        self.assertIn("serve", cli.SUBCOMMANDS)
        self.assertIsNone(args.db)

    def test_cli_serve_passes_db_and_name(self):
        """`serve --db X --name Y` 要真的传进去，否则默认库静默变成 data/sales.db。"""
        seen = {}
        with patch("app.mcp_server.serve", side_effect=lambda db, name: seen.update(
                db=db, name=name) or 0):
            args = cli.build_parser().parse_args(["serve", "--db", "x.db", "--name", "my"])
            self.assertEqual(cli.run_serve(args), 0)
        self.assertEqual(seen, {"db": "x.db", "name": "my"})

    def test_cli_serve_uses_default_name(self):
        seen = {}
        with patch("app.mcp_server.serve", side_effect=lambda db, name: seen.update(
                db=db, name=name) or 0):
            cli.run_serve(cli.build_parser().parse_args(["serve"]))
        self.assertEqual(seen["name"], SERVER_NAME)

    @unittest.skipUnless(HAS_MCP, NEEDS_MCP)
    def test_server_class_supports_both_mcp_generations(self):
        from app.mcp_server import _server_class
        cls = _server_class()
        for attr in ("tool", "run"):
            self.assertTrue(hasattr(cls, attr), attr)


class TestPayloadShared(unittest.TestCase):
    def test_cli_json_uses_same_builder(self):
        """CLI 的 `-f json` 与 MCP 工具必须是同一份结构，否则两边字段会各自演化。"""
        state = fake_state()
        self.assertEqual(cli._build_payload(state, "q"),
                         payload_mod.build_payload(state, "q"))

    def test_default_row_limit_is_sane(self):
        self.assertGreaterEqual(MAX_ROWS_DEFAULT, 50)
        self.assertLessEqual(MAX_ROWS_DEFAULT, 500)


if __name__ == "__main__":
    unittest.main()
