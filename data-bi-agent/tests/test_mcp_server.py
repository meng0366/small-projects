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
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from app import cli, payload as payload_mod
from app.mcp_server import (
    DEFAULT_HTTP_HOST, DEFAULT_HTTP_PATH, DEFAULT_HTTP_PORT,
    MAX_ROWS_DEFAULT,
    SERVER_NAME,
    ask_tool,
    build_http_app,
    build_server,
    catalog_tool,
    guard_http_app,
    normalize_http_config,
    schema_tool,
    serve,
    serve_http,
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


def _headers(host: str = "localhost:8000", auth: str = None):
    pairs = [(b"host", host.encode("latin-1"))]
    if auth is not None:
        pairs.append((b"authorization", auth.encode("utf-8")))
    return pairs


def _run_asgi(app, headers=(), scope_type: str = "http"):
    sent, called = [], []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": scope_type, "asgi": {"version": "3.0"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/mcp", "raw_path": b"/mcp", "query_string": b"",
        "headers": list(headers), "client": ("127.0.0.1", 1234),
        "server": ("test", 8000),
    }
    asyncio.run(app(scope, receive, send))
    return sent


def _status(messages):
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


def _body(messages):
    return b"".join(m.get("body", b"") for m in messages
                    if m["type"] == "http.response.body").decode("utf-8")


class TestHttpConfig(unittest.TestCase):
    def test_defaults_stay_on_loopback(self):
        cfg = normalize_http_config()
        self.assertEqual((cfg["host"], cfg["port"], cfg["path"]),
                         (DEFAULT_HTTP_HOST, DEFAULT_HTTP_PORT, DEFAULT_HTTP_PATH))
        self.assertFalse(cfg["remote"])
        self.assertFalse(cfg["read_only"])
        self.assertEqual(cfg["token"], "")
        self.assertIn("localhost", cfg["allowed_hosts"])

    def test_remote_without_token_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            normalize_http_config(host="0.0.0.0")
        self.assertIn("AGENT_MCP_TOKEN", str(cm.exception))

    def test_short_remote_token_is_refused(self):
        with self.assertRaises(ValueError):
            normalize_http_config(host="192.168.1.9", token="abc")

    def test_remote_with_token_defaults_read_only(self):
        cfg = normalize_http_config(host="0.0.0.0", token="s" * 20,
                                    allowed_hosts=["mcp.internal:9000, [::1]:1"])
        self.assertTrue(cfg["remote"])
        self.assertTrue(cfg["read_only"])
        self.assertIn("mcp.internal", cfg["allowed_hosts"])
        self.assertIn("::1", cfg["allowed_hosts"])

    def test_allow_write_turns_off_remote_read_only(self):
        cfg = normalize_http_config(host="0.0.0.0", token="s" * 20,
                                    allow_write=True)
        self.assertFalse(cfg["read_only"])

    def test_loopback_with_token_still_writes_by_default(self):
        cfg = normalize_http_config(host="127.0.0.1", token="s" * 20)
        self.assertFalse(cfg["remote"])
        self.assertFalse(cfg["read_only"])

    def test_bad_port_path_and_host(self):
        for kwargs in ({"port": 0}, {"port": "x"}, {"path": "mcp"},
                       {"path": "/mcp bad"}, {"host": "  "}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                normalize_http_config(**kwargs)


class TestHttpGuard(unittest.TestCase):
    def _fake_app(self, calls):
        async def app(scope, receive, send):
            calls.append(scope)
            await send({"type": "http.response.start", "status": 200,
                        "headers": []})
        return app

    def test_host_whitelist_blocks_dns_rebinding(self):
        calls = []
        app = guard_http_app(self._fake_app(calls),
                             allowed_hosts={"localhost", "127.0.0.1"})
        self.assertEqual(_status(_run_asgi(app, _headers("localhost:8000"))), 200)
        bad = _run_asgi(app, _headers("evil.example"))
        self.assertEqual(_status(bad), 403)
        self.assertIn("Host", _body(bad))
        self.assertEqual(len(calls), 1)

    def test_missing_or_wrong_bearer_returns_401_without_token_leak(self):
        calls = []
        token = "s" * 24
        app = guard_http_app(self._fake_app(calls), token=token,
                             allowed_hosts={"localhost"})
        for auth in (None, "Basic abc", "Bearer wrong", f"bearer {token}x"):
            with self.subTest(auth=auth):
                sent = _run_asgi(app, _headers("localhost", auth))
                self.assertEqual(_status(sent), 401)
                self.assertNotIn(token, _body(sent))
                start = next(m for m in sent if m["type"] == "http.response.start")
                self.assertIn((b"www-authenticate", b"Bearer"), start["headers"])
        self.assertEqual(calls, [])

    def test_valid_bearer_passes_through(self):
        calls = []
        token = "s" * 24
        app = guard_http_app(self._fake_app(calls), token=token,
                             allowed_hosts={"localhost"})
        sent = _run_asgi(app, _headers("localhost", f"Bearer {token}"))
        self.assertEqual(_status(sent), 200)
        self.assertEqual(len(calls), 1)

    def test_unicode_token_is_compared_as_bytes(self):
        """compare_digest 对 str 只收 ASCII；令牌里出现中文不该让服务 500。"""
        calls = []
        token = "令牌" * 10
        app = guard_http_app(self._fake_app(calls), token=token,
                             allowed_hosts={"localhost"})
        self.assertEqual(_status(_run_asgi(
            app, _headers("localhost", f"Bearer {token}"))), 200)
        self.assertEqual(_status(_run_asgi(
            app, _headers("localhost", "Bearer 错令牌"))), 401)
        self.assertEqual(len(calls), 1)

    def test_non_http_scope_is_untouched(self):
        calls = []
        app = guard_http_app(self._fake_app(calls), token="s" * 24)
        _run_asgi(app, _headers("evil.example"), scope_type="lifespan")
        self.assertEqual(len(calls), 1)

    def test_build_http_app_wraps_sdk_app_and_filters_kwargs(self):
        cfg = normalize_http_config(host="127.0.0.1", path="/rpc")

        class FakeServer:
            def __init__(self):
                self.kwargs = None

            def streamable_http_app(self, streamable_http_path=None):
                self.kwargs = {"streamable_http_path": streamable_http_path}

                async def app(scope, receive, send):
                    await send({"type": "http.response.start", "status": 200,
                                "headers": []})

                return app

        server = FakeServer()
        app = build_http_app(server, cfg)
        self.assertEqual(server.kwargs, {"streamable_http_path": "/rpc"})
        self.assertEqual(_status(_run_asgi(app, _headers("localhost"))), 200)
        self.assertEqual(_status(_run_asgi(app, _headers("evil.example"))), 403)

    def test_serve_http_hands_app_to_uvicorn(self):
        cfg = normalize_http_config(host="127.0.0.1", port=8123)

        class FakeServer:
            def streamable_http_app(self, **kwargs):
                async def app(scope, receive, send):
                    pass
                return app

        # 不假设测试环境装了 uvicorn：容器作业只装 requirements.txt。
        fake_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn}):
            self.assertEqual(serve_http(FakeServer(), cfg), 0)
        self.assertEqual(fake_uvicorn.run.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(fake_uvicorn.run.call_args.kwargs["port"], 8123)


class TestHttpServeDispatch(unittest.TestCase):
    def test_http_serve_builds_read_only_server_and_config(self):
        with patch("app.mcp_server.build_server") as build, \
                patch("app.mcp_server.serve_http", return_value=0) as http:
            code = serve(transport="streamable-http", host="0.0.0.0",
                         token="s" * 20)
        self.assertEqual(code, 0)
        self.assertTrue(build.call_args.kwargs["read_only"])
        self.assertTrue(http.call_args.args[1]["remote"])
        self.assertNotIn("run", http.call_args.kwargs)

    def test_unknown_transport_reports_on_stderr(self):
        err, out = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            code = serve(transport="ftp")
        self.assertEqual(code, 1)
        self.assertIn("未知传输方式", err.getvalue())
        self.assertEqual(out.getvalue(), "")

    def test_cli_http_flags_reach_serve(self):
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return 0

        argv = ["serve", "--transport", "streamable-http", "--host", "0.0.0.0",
                "--port", "9000", "--path", "/rpc", "--token", "s" * 20,
                "--allowed-host", "a.com,b.net", "--allow-write"]
        with patch("app.mcp_server.serve", side_effect=fake):
            self.assertEqual(cli.run_serve(cli.build_parser().parse_args(argv)), 0)
        self.assertEqual(seen["transport"], "streamable-http")
        self.assertEqual((seen["host"], seen["port"], seen["path"]),
                         ("0.0.0.0", 9000, "/rpc"))
        self.assertEqual(seen["allowed_hosts"], ["a.com,b.net"])
        self.assertTrue(seen["allow_write"])

    def test_cli_http_env_fallback(self):
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return 0

        env = {"AGENT_MCP_HOST": "localhost", "AGENT_MCP_PORT": "9111",
               "AGENT_MCP_PATH": "/env-mcp", "AGENT_MCP_TOKEN": "e" * 20,
               "AGENT_MCP_ALLOWED_HOSTS": "env.internal"}
        with patch.dict(os.environ, env), patch("app.mcp_server.serve", side_effect=fake):
            cli.run_serve(cli.build_parser().parse_args(
                ["serve", "--transport", "streamable-http"]))
        self.assertEqual((seen["host"], seen["port"], seen["path"]),
                         ("localhost", "9111", "/env-mcp"))
        self.assertEqual(seen["token"], "e" * 20)
        self.assertEqual(seen["allowed_hosts"], ["env.internal"])

    @unittest.skipUnless(HAS_MCP, NEEDS_MCP)
    def test_read_only_server_hides_catalog_and_rejects_chart(self):
        server = build_server(read_only=True)
        self.assertEqual([t.name for t in asyncio.run(server.list_tools())],
                         ["ask", "get_schema"])
        result = asyncio.run(server.call_tool("ask", {"question": "q", "chart": True}))
        self.assertIn("只读", str(result))


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
