"""MCP server：把 ask / schema / catalog 暴露给别的 Agent（P2，2026-11-26）。

为什么做这件事：这个项目的三项能力（问答、看结构、生成词典）都是给人用的 ——
命令行、网页。但"数据分析"最常见的调用方其实是另一个 Agent：写报告的 Agent
想知道本月销售额，它不该自己写 SQL，也不该让人复制粘贴。MCP 就是那根线。

三条硬约束，都是踩过或想过才写下来的：

1. **stdio 传输下 stdout 是协议通道**。任何一行 print 都会把 JSON-RPC 握手搅乱，
   客户端只看到"服务器启动失败"。所以这个模块里**一行业务 print 都没有**：
   工具只返回字符串，由 MCP 框架负责写 stdout。CLI 那套 `run_*` 函数全带 print，
   一个都不能直接调（这是复用边界，不是偷懒）。
2. **返回值会进另一个模型的上下文**。所以要截断数据（`MAX_ROWS_DEFAULT`）、
   要明确标注 `truncated`，并且**错误必须是结构化 JSON**而不是抛异常 ——
   对面看到一屏 traceback 只会说"工具坏了"，看到 `{"ok": false, "error": "..."}`
   才会换个问法或告诉用户原因。
3. **没装 mcp 不能让整个包 import 失败**。mcp 是可选依赖（它拖着 cryptography、
   pywin32 一堆东西），所以只在 `build_server()` 里延迟导入；纯函数
   `ask_tool` / `schema_tool` / `catalog_tool` 不依赖 mcp，可以离线测。

版本兼容：mcp 1.x 叫 `FastMCP`，2.x 改名 `MCPServer`。`tool()` 一致，但 HTTP
配置的位置变了（1.x 在构造函数、2.x 在 `run()` / `streamable_http_app()`），
所以用一层「只传目标版本认识的参数」的薄兼容，不复制两套实现。

0.17 加 HTTP 传输时守住一条线：stdio 是本机进程，谁起谁就是用户；HTTP 一旦
跨机器就不再有这层天然边界。因此默认只绑 127.0.0.1，非本机监听强制 Bearer
令牌（至少 16 字符）与 Host 白名单，且默认不注册会写盘的 `generate_catalog`。
"""
import hmac
import inspect
import ipaddress
import os
import sys
import time
from types import SimpleNamespace

from app.context import AgentContext
from app.payload import build_payload, dumps, schema_payload
from app.tools.catalog_gen import (
    build_draft,
    count_todos,
    describe_changes,
    generate_descriptions,
    parse_draft,
    profile_schema,
    render_catalog_yaml,
    summarize_profile,
    validate_draft,
)
from app.schema.loader import load_catalog

SERVER_NAME = "data-bi-agent"
MAX_ROWS_DEFAULT = 200
"""单次返回的行数上限。200 行足够看清分布，又不至于把对面的上下文塞满。"""

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_HTTP_PATH = "/mcp"
MIN_REMOTE_TOKEN_LEN = 16
"""非本机监听的令牌下限：短到能被猜出来的 token 等于没设。"""

INSTRUCTIONS = """\
这是「数据分析 Agent」：用中文自然语言提问，它把问题翻译成 SQL 在只读库上执行，
返回结果数据、生成的语句与一句中文结论。

- `ask`：问一个数据问题。问题要用中文，说清指标、维度、时间范围。
  它只读、不会改库；拿不到结果时会说明原因（该澄清、该拒答都不编数）。
- `get_schema`：先看库里有哪些表、哪些列、有哪些指标口径。不确定表结构时先调这个。
- `generate_catalog`：给陌生的库生成业务词典草稿（会写文件，需要人工确认后才生效）。

返回都是 JSON。`data` 可能被截断，看 `truncated` 与 `rows`；`ok=false` 时看 `error`。
"""


# ---------------------------------------------------------------- HTTP 边界（纯函数，可离线测）
def _host_only(value: str) -> str:
    """把 ``example.com:8000`` / ``[::1]:8000`` 归一成裸主机名。"""
    text = (value or "").strip()
    if text.startswith("["):
        return text[1:].split("]", 1)[0].strip().lower()
    if text.count(":") == 1:
        return text.rsplit(":", 1)[0].strip().lower()
    return text.lower()


def _is_loopback(host: str) -> bool:
    name = _host_only(host)
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def normalize_http_config(host: str = DEFAULT_HTTP_HOST,
                          port: int = DEFAULT_HTTP_PORT,
                          path: str = DEFAULT_HTTP_PATH, token: str = "",
                          allowed_hosts=(), allow_write: bool = False) -> dict:
    """校验并归一 HTTP 启动参数；不合法时抛 ValueError（由 serve 打到 stderr）。

    安全默认：本机绑定、非本机必须带够长的令牌；本机也校验 Host 头，
    挡住 DNS rebinding（浏览器把 127.0.0.1 当成自己内网的一块跳板）。
    """
    host = str(host or DEFAULT_HTTP_HOST).strip()
    if not host or any(ch.isspace() for ch in host):
        raise ValueError("--host 不能为空或包含空格。")
    try:
        port = int(port)
    except (TypeError, ValueError) as e:
        raise ValueError("--port 必须是 1–65535 的整数。") from e
    if not 1 <= port <= 65535:
        raise ValueError(f"--port 超出范围：{port}（要是 1–65535）。")
    path = str(path or DEFAULT_HTTP_PATH).strip()
    if not path.startswith("/") or any(ch.isspace() for ch in path):
        raise ValueError("--path 必须以 / 开头且不能含空格（如 /mcp）。")

    token = (token or "").strip()
    remote = not _is_loopback(host)
    if remote and len(token) < MIN_REMOTE_TOKEN_LEN:
        raise ValueError(
            f"非本机监听（--host {host}）必须设置至少 {MIN_REMOTE_TOKEN_LEN} 字符的"
            " Bearer 令牌：--token 或环境变量 AGENT_MCP_TOKEN。"
            " 只监听 127.0.0.1 可不带令牌。"
        )

    hosts = {"localhost", "127.0.0.1", "::1", _host_only(host)}
    for item in allowed_hosts or []:
        for part in str(item).split(","):
            if part.strip():
                hosts.add(_host_only(part))
    hosts.discard("")
    return {
        "host": host, "port": port, "path": path, "token": token,
        "allowed_hosts": frozenset(hosts), "remote": remote,
        # 跨机器默认只读：generate_catalog 会写文件，chart=true 也会落图片。
        "read_only": remote and not allow_write,
    }


async def _send_http_error(send, status: int, message: str,
                           extra_headers=()) -> None:
    """鉴权失败也返回结构化 JSON，跟工具错误保持一种形状。"""
    body = dumps({"ok": False, "error": message}).encode("utf-8")
    headers = [(b"content-type", b"application/json; charset=utf-8"),
               *extra_headers]
    await send({"type": "http.response.start", "status": status,
                "headers": headers})
    await send({"type": "http.response.body", "body": body})


def guard_http_app(app, *, token: str = "", allowed_hosts=()):
    """给 MCP 的 ASGI app 套上 Bearer 令牌与 Host 白名单。

    mcp SDK 自带的 transport_security 在 1.x / 2.x 之间换过位置，自己包一层
    ASGI 反而更稳；而且这样令牌校验是**所有请求之前**的第一道门，不会漏掉
    `/mcp` 之外的路径。
    """
    expected = (token or "").strip()
    allowed = frozenset(_host_only(h) for h in (allowed_hosts or ()))

    async def guarded(scope, receive, send):
        if scope.get("type") not in ("http", "websocket"):
            await app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("utf-8", "replace")
            for k, v in scope.get("headers", [])
        }
        if allowed:
            incoming = _host_only(headers.get("host", ""))
            if incoming not in allowed:
                await _send_http_error(
                    send, 403, f"Host 不被允许：{incoming or '(空)'}"
                               "（启动时用 --allowed-host 添加）")
                return
        if expected:
            scheme, _, supplied = headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                    supplied.strip().encode("utf-8"), expected.encode("utf-8")):
                await _send_http_error(
                    send, 401, "需要 Bearer 令牌（Authorization: Bearer <token>）",
                    [(b"www-authenticate", b"Bearer")])
                return
        await app(scope, receive, send)

    return guarded


def _supported_kwargs(target, candidates: dict) -> dict:
    """只传目标版本认识的参数——1.x / 2.x 的 HTTP 配置位置不同。"""
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return candidates
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidates
    return {k: v for k, v in candidates.items() if k in params}


# ---------------------------------------------------------------- 工具实现（纯函数，不依赖 mcp）
def _ask_namespace(db: str, plan: bool, peek: bool, insight: bool,
                   chart: bool, linear: bool) -> SimpleNamespace:
    """把工具参数翻译成 CLI 认识的那套开关。

    复用 `cli.ask_once`（它内部再走 `build_agent`）而不是自己拼引擎：
    引擎装配、图表/洞察开关、追踪号这些逻辑在 CLI 与网页已经有两处调用，
    再抄第三份必然漂移。这里只是把参数名字对齐。
    """
    return SimpleNamespace(
        db=db or None,
        no_plan=not plan,
        no_peek=not peek,
        no_chart=not chart,
        no_insight=not insight,
        linear=linear,
        pandas=False,
        format="json",
        quiet=True,
        out=None,
    )


def _fail(message: str) -> str:
    return dumps({"ok": False, "error": message})


def ask_tool(question: str, db: str = "", plan: bool = True, peek: bool = True,
             insight: bool = False, chart: bool = False, linear: bool = False,
             max_rows: int = MAX_ROWS_DEFAULT) -> str:
    """问一个数据问题，返回 JSON 字符串。

    图表默认关闭：MCP 场景下调用方要的是数字，不是一张图文件。
    洞察默认关闭：多一次模型调用，需要中文结论时显式打开。
    """
    from app.cli import ask_once          # 延迟导入：避免 mcp_server ←→ cli 的顶层循环

    text = (question or "").strip()
    if not text:
        return _fail("question 不能为空，请描述你想看的数据（中文自然语言）。")

    if db:
        path = os.path.abspath(os.path.expanduser(db))
        if not os.path.exists(path):
            # 与命令行一致：库文件不存在就明说，不静默降级成 CSV 模式
            return _fail(f"数据库文件不存在：{path}")

    limit = 0 if max_rows is None else max(0, int(max_rows))
    try:
        state = ask_once(_ask_namespace(db, plan, peek, insight, chart, linear), text)
    except Exception as e:                # 网络 / 密钥 / 模型异常：转成可读错误
        return _fail(f"{type(e).__name__}: {e}")

    payload = build_payload(state, text, max_rows=limit)
    if not payload["ok"]:
        # `answer` 里是给人看的原因（澄清提示 / 拒答理由 / 执行报错）
        payload["error"] = (state.get("answer") or "").strip() or "未拿到结果。"
    return dumps(payload)


def schema_tool(db: str = "") -> str:
    """返回库结构与指标口径（JSON）。"""
    ctx = AgentContext(db_file=db or None)
    if db and not ctx.has_db:
        return _fail(f"数据库文件不存在：{os.path.abspath(os.path.expanduser(db))}")
    try:
        payload = schema_payload(ctx)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")
    payload["ok"] = True
    return dumps(payload)


def catalog_tool(db: str, out: str = "", merge: str = "", dry_run: bool = False) -> str:
    """给一个陌生的库生成 catalog 草稿，返回草稿路径与问题清单（JSON）。

    这是**唯一有写操作**的工具，所以：默认写到库文件同目录的 `catalog.draft.yaml`，
    绝不覆盖 `app/schema/catalog.yaml`；`merge` 只补空；`dry_run` 完全不写盘也不调模型。
    """
    if not db:
        return _fail("必须指定 db（SQLite 库文件路径）。")
    path = os.path.abspath(os.path.expanduser(db))
    if not os.path.exists(path):
        return _fail(f"数据库文件不存在：{path}")

    try:
        profile = profile_schema(path)
    except Exception as e:
        return _fail(f"读取库结构失败：{type(e).__name__}: {e}")
    if not profile["tables"]:
        return dumps({"ok": False, "error": "这个库里没有可读的表。",
                      "notes": profile.get("notes") or []})

    base = {"ok": True, "db": path, "summary": summarize_profile(profile),
            "notes": profile.get("notes") or []}
    if dry_run:
        base.update({"dry_run": True, "tables": profile["tables"]})
        return dumps(base)

    existing = None
    if merge:
        existing = load_catalog(os.path.abspath(os.path.expanduser(merge)))
        if not existing:
            return _fail(f"找不到可合并的词典（或文件是空的）：{merge}")

    try:
        llm_raw = generate_descriptions(profile)
    except Exception as e:
        return _fail(f"调用模型失败：{type(e).__name__}: {e}")
    llm_draft = parse_draft(llm_raw)
    if llm_draft is None:
        return _fail("模型返回的不是可解析的 JSON，未生成草稿（没写文件）。"
                     f" 原始输出前 300 字：{llm_raw[:300]}")

    issues = validate_draft(llm_draft, profile)
    draft = build_draft(profile, llm_draft, existing)

    out_path = out or os.path.join(os.path.dirname(path), "catalog.draft.yaml")
    out_abs = os.path.abspath(os.path.expanduser(out_path))
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    text = render_catalog_yaml(
        draft, source=path,
        generated_at=time.strftime("%Y-%m-%d %H:%M"),
        notes=[f"模型提到的 {i['message'][:30]} 已丢弃"
               for i in issues if i.get("level") == "error"][:3],
    )
    with open(out_abs, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    return dumps({
        **base,
        "path": out_abs,
        "todos": count_todos(draft),
        "changes": describe_changes(draft, existing),
        "issues": issues,
        "dropped": [i["message"] for i in issues if i.get("level") == "error"],
        "next": "草稿需要人工确认：搜 TODO 补齐中文名/说明，再覆盖 app/schema/catalog.yaml"
                "（或用 AGENT_CATALOG 指向它先试用）。",
    })


# ---------------------------------------------------------------- 组装 server
def _server_class():
    """取 MCP 服务器类，兼容 1.x（FastMCP）与 2.x（改名为 MCPServer）。"""
    try:
        from mcp.server.mcpserver import MCPServer as cls          # mcp >= 2
        return cls
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP as cls              # mcp 1.x
        return cls
    except ImportError as e:
        raise ModuleNotFoundError(
            "没有安装 mcp。MCP 是可选能力：pip install -r requirements-mcp.txt"
        ) from e


def build_server(db: str = "", name: str = SERVER_NAME, *,
                 read_only: bool = False, http_config: dict = None):
    """构建 MCP server。`db` 作为默认库，工具参数里仍可逐次覆盖。

    `read_only=True` 时不注册 `generate_catalog`，并拒绝 `ask(chart=True)` ——
    跨机器访问没有本地进程那层天然边界，写盘工具要显式开启。
    """
    Server = _server_class()
    options = {"name": name, "instructions": INSTRUCTIONS}
    if http_config:
        # 1.x 把 HTTP 参数放构造函数，2.x 放 streamable_http_app()；两边都不认识就忽略。
        options.update(_supported_kwargs(Server, {
            "streamable_http_path": http_config["path"],
            "host": http_config["host"],
        }))
    server = Server(**options)
    default_db = db or ""

    def ask(question: str, db: str = default_db, plan: bool = True, peek: bool = True,
            insight: bool = False, chart: bool = False, linear: bool = False,
            max_rows: int = MAX_ROWS_DEFAULT) -> str:
        """用中文自然语言问一个数据问题，返回结果数据、生成的 SQL 与执行信息。

        只读：不会修改库。拿不到结果时会说明原因（该澄清或该拒答时不会编数字）。
        plan / peek 是"先出查询计划"与"先看一眼真实取值"两项自检，默认开启；
        想省钱可以关掉。chart / insight 默认关闭（它们会多花 token 或写图片文件）。
        """
        if read_only and chart:
            return _fail("当前是只读 MCP 服务：chart=true 会写图片文件，已拒绝。"
                         " 如需生成图片，请用本机 stdio 启动，或显式 --allow-write。")
        return ask_tool(question, db=db, plan=plan, peek=peek,
                        insight=insight, chart=chart, linear=linear, max_rows=max_rows)

    def get_schema(db: str = default_db) -> str:
        """查看数据源结构：有哪些表、每张表有哪些列（含中文名）、有哪些指标口径与表关系。

        不确定表名或列名时先调这个，再提问。
        """
        return schema_tool(db=db)

    def generate_catalog(db: str, out: str = "", merge: str = "",
                         dry_run: bool = False) -> str:
        """为一个陌生的 SQLite 库生成业务语义词典（catalog）草稿。

        事实（类型/样例/枚举取值/外键/行数）由代码实测，中文名与口径由模型提议，
        最后落成带 TODO 的 YAML 草稿等人工确认。**会写文件**：默认写到库文件同目录
        的 catalog.draft.yaml，不会覆盖项目内置的 app/schema/catalog.yaml。
        dry_run=true 时只做代码侧识别、不调模型、不写盘。
        """
        return catalog_tool(db=db, out=out, merge=merge, dry_run=dry_run)

    server.tool(name="ask", description=ask.__doc__)(ask)
    server.tool(name="get_schema", description=get_schema.__doc__)(get_schema)
    if not read_only:
        server.tool(name="generate_catalog",
                    description=generate_catalog.__doc__)(generate_catalog)
    return server


def build_http_app(server, config: dict):
    """把 MCP server 变成带鉴权的 ASGI app，供 uvicorn 直接跑。"""
    factory = getattr(server, "streamable_http_app", None)
    if not callable(factory):
        raise RuntimeError(
            "当前 mcp 版本没有 streamable_http_app()，请装 mcp>=1.9："
            "pip install -r requirements-mcp.txt")
    # 2.x 的 app builder 收 path/host；1.x 在构造时已经设过，这里过滤掉就好。
    app = factory(**_supported_kwargs(factory, {
        "streamable_http_path": config["path"],
        "host": config["host"],
    }))
    return guard_http_app(app, token=config["token"],
                          allowed_hosts=config["allowed_hosts"])


def serve_http(server, config: dict) -> int:
    """启动 streamable-http。uvicorn 是 mcp 的传递依赖，仍延迟导入。"""
    import uvicorn                      # noqa: F401——只在真要起 HTTP 时才需要
    app = build_http_app(server, config)
    uvicorn.run(app, host=config["host"], port=config["port"],
                log_level="warning", access_log=False)
    return 0


def serve(db: str = "", name: str = SERVER_NAME, *,
          transport: str = "stdio", host: str = DEFAULT_HTTP_HOST,
          port: int = DEFAULT_HTTP_PORT, path: str = DEFAULT_HTTP_PATH,
          token: str = "", allowed_hosts=(), allow_write: bool = False) -> int:
    """启动 MCP server：stdio（默认）或 streamable-http。"""
    try:
        if transport == "stdio":
            server = build_server(db=db, name=name)
            server.run("stdio")
            return 0
        if transport != "streamable-http":
            raise ValueError(f"未知传输方式：{transport}（只支持 stdio / streamable-http）")
        config = normalize_http_config(host=host, port=port, path=path,
                                       token=token, allowed_hosts=allowed_hosts,
                                       allow_write=allow_write)
        server = build_server(db=db, name=name, read_only=config["read_only"],
                              http_config=config)
        return serve_http(server, config)
    except (ModuleNotFoundError, ValueError, RuntimeError) as e:
        print(f"[启动失败] {e}", file=sys.stderr)
        return 1
