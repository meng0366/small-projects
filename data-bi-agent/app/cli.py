"""命令行入口（阶段 6：子命令化 + 脚本友好）。

用法：

    python -m app.cli                      # 交互问答（默认行为，同 chat）
    python -m app.cli "各区域的销售额是多少？"     # 单次提问（旧写法仍可用，等价 ask）

    python -m app.cli ask "各区域的销售额是多少？" -f json
    python -m app.cli ask "各品类的销售额是多少？" -o 品类销售额.xlsx
    echo "各区域的销售额是多少？" | python -m app.cli ask -
    python -m app.cli chat -c              # 接着上一次会话继续问
    python -m app.cli chat --list-sessions # 列出本地会话
    python -m app.cli schema               # 看数据源、表结构、字段释义、指标口径
    python -m app.cli logs -n 5 --failed   # 看最近 5 条失败的运行记录
    python -m app.cli logs --trace 20261124-225251-cb5d   # 看某一次的各阶段耗时
    python -m app.cli teach "问题" --sql "SELECT ..."   # 把答错的问题教成 few-shot
    python -m app.cli lessons              # 查看 / 删除已沉淀的修正记录
    python -m app.cli catalog --db 我的库.db          # 给陌生的库自动生成语义词典草稿
    python -m app.cli catalog --db 我的库.db --dry-run  # 只看代码能自动认出什么（零成本）

设计取舍（借鉴来源见 docs/参考项目记录.md 第 7–9 节）：
- **只用标准库 argparse**：项目已经带了 LangChain 一大串依赖，CLI 层不再引入
  typer / rich / click，clone 下来就能跑；
- **输出分两类**：`table` 给人看（带路由、生成的语句、洞察、追问建议）；
  `json` / `csv` / `markdown` **只输出数据本身**，方便 `> out.csv`、`| jq`、粘进文档；
- **退出码有含义**：0 成功、1 分析失败、2 参数或数据源问题，shell 里可直接判断；
- **写盘默认 utf-8-sig**（见 app/render.py）：Windows 上 Excel 双击打开不乱码。
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

from app import sessions
from app.agent import DataAgent
from app.config import LLM_MODEL
from app.context import AgentContext
from app.graph import GraphAgent
from app.logging_utils import log_path, read_events, read_runs
from app.payload import build_payload, schema_payload
from app.render import (
    SUPPORTED_FORMATS,
    UnsupportedFormat,
    export,
    normalize_format,
    render,
)
from app.schema.loader import list_tables, load_catalog
from app.tools.chart_tool import CHART_TYPE_LABEL
from app.tools.catalog_gen import (
    CatalogError,
    build_draft,
    count_todos,
    describe_changes,
    format_profile,
    generate_descriptions,
    parse_draft,
    profile_schema,
    render_catalog_yaml,
    summarize_profile,
    validate_draft,
)
from app.tools.lesson_store import (
    LessonError,
    add_lesson,
    db_fingerprint,
    lessons_path,
    load_lessons,
    remove_lesson,
    score_lesson,
    select_lessons,
)
from app.tools.plan_guard import format_plan_summary

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文打印报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

VERSION = "0.14.0"
"""与项目阶段对齐：0.10 = 计划先行；0.11 = 两段式探查；0.12 = 词典描述自动生成；
0.13 = MCP server（把 ask / schema / catalog 暴露给别的 Agent）；
0.13.1 = 半年（上半年 / H1）纳入范围守卫，并修掉「今年上半年」被当成全年的锚点误判；
0.14 = 提示词去重：schema / 术语只注入一次（输入 token -49.4%），行为不变。"""

SUBCOMMANDS = ("ask", "a", "chat", "c", "schema", "s", "logs", "l",
               "teach", "t", "lessons", "ls", "catalog", "cat",
               "serve", "version", "help")
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2

ROUTE_LABEL = {
    "sql": "SQLite 多表（Text-to-SQL）",
    "pandas": "单表宽表（Text-to-Pandas）",
}


# ---------------------------------------------------------------- 参数
def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器（单独暴露，便于测试）。"""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="数据分析 Agent 命令行：用自然语言提问，自动生成 SQL/代码并执行。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m app.cli ask \"各区域的销售额是多少？\"\n"
            "  python -m app.cli ask \"各品类的销售额是多少？\" -f csv -o 品类.csv\n"
            "  python -m app.cli chat -c\n"
            "  python -m app.cli schema\n"
            "  python -m app.cli logs -n 5 --failed\n"
            "  python -m app.cli teach \"各城市的销售额是多少？\" --sql \"SELECT ...\"\n"
            "  python -m app.cli catalog --db 我的库.db --dry-run\n"
            "  python -m app.cli serve            # 以 MCP server 启动，给 Claude Desktop / IDE 用\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    def add_engine_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", metavar="PATH", help="使用指定的 SQLite 库文件（默认 data/sales.db）")
        p.add_argument("--no-chart", action="store_true", help="不生成图表")
        p.add_argument("--no-insight", action="store_true", help="不生成洞察文案（省一次模型调用）")
        p.add_argument("--no-plan", action="store_true",
                       help="不先出查询计划（省一次模型调用；计划用于提前发现理解错误）")
        p.add_argument("--no-peek", action="store_true",
                       help="不允许两段式探查（模型不确定取值时先看一眼真实数据）")
        p.add_argument("--linear", action="store_true", help="用线性编排（无重试、无多轮记忆）")
        p.add_argument("--pandas", action="store_true", help="线性模式下强制走单表 pandas")

    def add_format_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("-f", "--format", default="table", metavar="FMT",
                       help=f"输出格式：{' / '.join(SUPPORTED_FORMATS)}（默认 table）")
        p.add_argument("-o", "--out", metavar="PATH",
                       help="把结果导出到文件，按扩展名识别：.csv / .json / .md / .xlsx")
        p.add_argument("-q", "--quiet", action="store_true",
                       help="不打印生成的 SQL / 代码与过程信息")

    ask = sub.add_parser("ask", aliases=["a"], help="一次提问并输出结果")
    ask.add_argument("question", nargs="?", help="自然语言问题；写 - 表示从标准输入读取")
    add_format_options(ask)
    add_engine_options(ask)

    chat = sub.add_parser("chat", aliases=["c"], help="交互式问答（支持追问与续聊）")
    chat.add_argument("-c", "--continue", dest="resume", action="store_true",
                      help="接着上一次会话继续（等价于接着最近一次）")
    chat.add_argument("--session", metavar="NAME", help="指定会话名（存在则接着聊）")
    chat.add_argument("--list-sessions", action="store_true", help="列出本地保存的会话")
    add_engine_options(chat)

    schema_cmd = sub.add_parser("schema", aliases=["s"], help="查看数据源、表结构、字段释义与指标口径")
    schema_cmd.add_argument("--json", action="store_true", help="以 JSON 输出（脚本用）")
    schema_cmd.add_argument("--db", metavar="PATH", help="使用指定的 SQLite 库文件")

    logs_cmd = sub.add_parser("logs", aliases=["l"], help="查看运行记录（耗时、token、失败原因）")
    logs_cmd.add_argument("-n", type=int, default=10, metavar="N", help="显示最近 N 条（默认 10）")
    logs_cmd.add_argument("--failed", action="store_true", help="只看失败的记录")
    logs_cmd.add_argument("-t", "--trace", metavar="ID", default="",
                          help="显示某次问答的完整阶段链路（支持追踪号前缀）")
    logs_cmd.add_argument("--json", action="store_true", help="以 JSON 输出")

    teach = sub.add_parser("teach", aliases=["t"],
                           help="把「问题 → 正确 SQL」存成 few-shot（修正回流）")
    teach.add_argument("question", help="答错的那个问题（原话）")
    teach.add_argument("-s", "--sql", dest="sql", required=True, metavar="SQL",
                       help="正确的 SQL；若这类问题该拒答，写 DEFER: 原因")
    teach.add_argument("-n", "--note", default="", help="备注：为什么这么改（只在本地展示）")
    teach.add_argument("--force", action="store_true",
                       help="跳过\"SQL 能否执行\"的校验（不推荐，存进去会教模型写错语句）")
    teach.add_argument("--db", metavar="PATH", help="使用指定的 SQLite 库文件")

    lessons = sub.add_parser("lessons", aliases=["ls"], help="查看/删除已沉淀的修正记录")
    lessons.add_argument("--rm", metavar="ID", help="删除指定 id 的记录")
    lessons.add_argument("--match", metavar="QUESTION",
                         help="看某个问题会命中哪几条记录（调试相关度用）")
    lessons.add_argument("--json", action="store_true", help="以 JSON 输出")
    lessons.add_argument("--db", metavar="PATH", help="使用指定的 SQLite 库文件")

    catalog_cmd = sub.add_parser(
        "catalog", aliases=["cat"],
        help="从一个陌生的库生成语义词典草稿（描述自动生成）")
    catalog_cmd.add_argument("--db", metavar="PATH", required=True,
                             help="要生成词典的 SQLite 库文件")
    catalog_cmd.add_argument("-o", "--out", metavar="PATH", default="",
                             help="草稿输出路径（默认当前目录下的 catalog.draft.yaml）")
    catalog_cmd.add_argument("--dry-run", action="store_true",
                             help="只做代码侧识别、不调模型（零成本，先看能自动认出什么）")
    catalog_cmd.add_argument("--merge", metavar="PATH",
                             help="与已有词典合并：已有的非空内容一律保留，只补空缺")
    catalog_cmd.add_argument("--json", action="store_true", help="以 JSON 输出（不写文件）")

    serve_cmd = sub.add_parser(
        "serve", help="以 MCP server 形态启动（stdio），供别的 Agent 直接调用")
    serve_cmd.add_argument("--db", metavar="PATH",
                           help="默认库文件；MCP 工具的参数里仍可逐次覆盖")
    serve_cmd.add_argument("--name", default="", metavar="NAME",
                           help="服务器名（默认 data-bi-agent）")

    sub.add_parser("version", help="显示版本与当前数据源")

    return parser


def normalize_argv(argv: list) -> list:
    """兼容旧写法：`python -m app.cli "问题" --no-chart` 视作 `ask`。"""
    if not argv:
        return ["chat"]
    first = argv[0]
    if first in SUBCOMMANDS or first in ("-h", "--help"):
        return list(argv)
    return ["ask", *argv]


# ---------------------------------------------------------------- 通用
def make_context(args) -> AgentContext:
    """按 `--db` 构造数据上下文。"""
    db_file = getattr(args, "db", None)
    use_plan = not getattr(args, "no_plan", False)
    use_peek = not getattr(args, "no_peek", False)
    return AgentContext(db_file=db_file, use_plan=use_plan, use_peek=use_peek)


def build_agent(args):
    """按参数构建 Agent，返回 (agent, 引擎说明)。"""
    ctx = make_context(args)
    with_chart = not getattr(args, "no_chart", False)
    with_insight = not getattr(args, "no_insight", False)
    if getattr(args, "linear", False):
        # 线性引擎也收 with_chart / with_insight / force_pandas，
        # 否则 `chat --linear --no-chart` 这类组合会被静默忽略
        return DataAgent(ctx=ctx, with_chart=with_chart, with_insight=with_insight,
                         force_pandas=getattr(args, "pandas", False)), \
            "线性编排（无重试、无记忆）"
    return GraphAgent(ctx, with_chart=with_chart, with_insight=with_insight), \
        "LangGraph 状态机（含重试与多轮记忆）"


def resolve_question(question: Optional[str]) -> str:
    """取问题：支持 `-` 或空参数时从标准输入读（管道友好）。"""
    if question is None or question == "-":
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
        return ""
    return question.strip()


def ask_once(args, question: str) -> dict:
    """执行一次问答，返回状态 dict（附 `_elapsed`）。

    两种引擎都走 `agent.run()`：线性版补了同签名接口（含追踪号），
    否则 `chat --linear` 会 AttributeError，`ask --linear` 也拿不到追踪号。
    """
    started = time.perf_counter()
    agent, _ = build_agent(args)
    state = agent.run(question, [])
    state["_elapsed"] = round(time.perf_counter() - started, 2)
    return state


# ---------------------------------------------------------------- 输出
def _print_trace_hint(state: dict) -> None:
    """追踪号放在最后：只有要排查时才需要它，不该占结论的位置。"""
    trace = state.get("trace_id") or ""
    if trace:
        print(f"\n[追踪] {trace}（看这次各阶段耗时："
              f"python -m app.cli logs --trace {trace}）")


def _print_human(state: dict, question: str, quiet: bool = False) -> None:
    """人类可读输出：过程信息 + 结果表 + 图表 + 洞察 + 追问建议。"""
    rewritten = state.get("rewritten_question") or ""
    if rewritten and rewritten != question:
        print(f"[改写] {rewritten}")

    route = state.get("route") or ""
    if route:
        print(f"[路由] {ROUTE_LABEL.get(route, route)}")

    retries = int(state.get("retry_count") or 0)
    if retries:
        print(f"[重试] 共重试 {retries} 次后完成")
        for err in state.get("errors") or []:
            print(f"  · {err}")

    df = state.get("result")
    if df is None:
        print(f"\n{state.get('answer') or '[未能回答]'}")
        for err in (state.get("errors") or [])[-1:]:
            print(f"  · {err}")
        _print_trace_hint(state)
        return

    # 计划先行（P1）：先把"模型理解成了什么"摆出来，再看它写了什么 SQL。
    # 只在真的出了计划时显示 —— 简单问题不走这步，多一行空标题没意义。
    plan = state.get("plan") or {}
    if plan:
        print(f"\n[计划] {format_plan_summary(plan)}")
        for issue in state.get("plan_issues") or []:
            if issue.get("kind") == "assumption":
                print(f"  · 假设：{issue.get('message')}")
            elif issue.get("level") == "warn" and issue.get("kind") in ("table", "column"):
                print(f"  · 注意：{issue.get('message')}")

    # 两段式探查（P1）：只在这轮真的探查过时出现 —— 它说明"模型对某个取值没把握，
    # 先看了真实数据再写 SQL"，用户能据此判断这个数字的可靠性。
    if state.get("peek_used"):
        peek_note = state.get("peek_note") or ""
        peek_error = state.get("peek_error") or ""
        if peek_note:
            print(f"\n[探查] {peek_note}")
        elif peek_error:
            print(f"\n[探查] 未成功：{peek_error}")

    statement = state.get("statement") or ""
    if statement and not quiet:
        label = "SQL" if route == "sql" else "代码"
        print(f"\n[生成的{label}]")
        print(statement)

    print(f"\n[分析结果]（{len(df)} 行）")
    if len(df) == 0:
        # 空表打印"Empty DataFrame…"对用户没意义，直接说明没有数据
        print("（当前条件下没有匹配的数据行）")
    else:
        print(render(df, "table"))

    # 归因：把分项变化单独列出来 —— 结论里只放了最大的三项，
    # 完整的加减表在这儿，方便逐项核对"总和到底对不对"。
    att = state.get("attribution") or {}
    if att.get("ok"):
        print(f"\n[归因] 「{att.get('current_label')}」对比「{att.get('base_label')}」，"
              f"按「{att.get('dim')}」拆解（指标：{att.get('metric')}）")
        print(render(att["table"], "table"))
        drill = att.get("drill")
        if drill:
            print(f"\n[归因·下钻] 把变化最大的「{drill.get('item')}」"
                  f"再按「{drill.get('dim')}」拆开")
            print(render(drill["table"], "table"))

    chart_path = state.get("chart_path") or ""
    chart_type = state.get("chart_type") or "none"
    if chart_path:
        print(f"\n[图表] {CHART_TYPE_LABEL.get(chart_type, chart_type)} → {chart_path}")

    if state.get("insight"):
        print(f"\n[洞察] {state['insight']}")

    suggestions = state.get("suggestions") or []
    if suggestions:
        print("\n[可以接着问]")
        for item in suggestions:
            print(f"  · {item}")

    _print_trace_hint(state)


def _build_payload(state: dict, question: str) -> dict:
    """命令行 `-f json` 的结构化输出。

    实现在 `app/payload.py` —— MCP 工具返回的是同一份结构，两边各写一份必然漂移。
    命令行不截断（用户自己要看全量），截断只在 MCP 侧按 `max_rows` 做。
    """
    return build_payload(state, question)


def output_result(args, state: dict, question: str) -> int:
    """按格式输出并处理导出，返回退出码。"""
    df = state.get("result")
    ok = df is not None
    fmt = normalize_format(getattr(args, "format", "table"))

    if fmt == "json":
        print(json.dumps(_build_payload(state, question), ensure_ascii=False,
                         indent=2, default=str))
    elif fmt in ("csv", "markdown"):
        # 管道 / 重定向场景：标准输出只放数据，别的都去 stderr
        if ok:
            print(render(df, fmt))
        else:
            print(state.get("answer") or "未能完成分析。", file=sys.stderr)
    else:
        _print_human(state, question, quiet=getattr(args, "quiet", False))

    out = getattr(args, "out", None)
    if out and ok:
        try:
            path = export(df, out)
        except UnsupportedFormat as e:
            print(f"[导出失败] {e}", file=sys.stderr)
            return EXIT_USAGE
        print(f"[已导出] {path}", file=sys.stderr)

    return EXIT_OK if ok else EXIT_FAILED


# ---------------------------------------------------------------- 子命令
def run_ask(args) -> int:
    question = resolve_question(args.question)
    if not question:
        print("请提供问题，例如：python -m app.cli ask \"各区域的销售额是多少？\"",
              file=sys.stderr)
        return EXIT_USAGE

    try:
        state = ask_once(args, question)
    except Exception as e:   # 网络 / API 异常：给出可读错误，而不是一屏堆栈
        print(f"[调用失败] {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_FAILED

    return output_result(args, state, question)


def _resolve_session(args) -> tuple:
    """返回 (会话名, 会话数据或 None)。"""
    if args.session:
        try:
            return args.session, sessions.load_session(args.session)
        except sessions.SessionError:
            return args.session, None      # 指定名字但还没有 → 新建同名会话
    if args.resume:
        try:
            session = sessions.load_session(None)
            return session["name"], session
        except sessions.SessionError as e:
            print(f"[提示] {e}，将开始新会话", file=sys.stderr)
    return sessions.new_id(), None


def print_sessions() -> int:
    rows = sessions.list_sessions()
    if not rows:
        print("还没有保存的会话。直接运行 `python -m app.cli chat` 就会新建一个。")
        return EXIT_OK

    print(f"会话目录：{sessions.sessions_dir()}")
    for row in rows:
        last = (row.get("last") or "").replace("\n", " ")[:40]
        print(f"· {row['name']}  {row.get('turns', 0)} 轮  {row.get('updated', '')}  {last}")
    print("\n接着聊：python -m app.cli chat -c   （或 --session 会话名）")
    return EXIT_OK


def _handle_slash(command: str, args) -> tuple:
    """处理交互模式里的 / 命令，返回 (是否继续循环, 是否清空历史)。"""
    if command in ("/quit", "/exit", "/q"):
        return False, False
    if command == "/help":
        print(
            "\n可用命令：\n"
            "  /help    显示本帮助\n"
            "  /schema  查看数据源与表结构\n"
            "  /logs    查看最近 5 条运行记录\n"
            "  /clear   清空上下文，开始新会话\n"
            "  /quit    退出\n"
            "\n直接输入问题即可，支持追问，例如先问「各区域销售额」，再问「只看华东」。"
        )
        return True, False
    if command == "/schema":
        run_schema(argparse.Namespace(json=False, db=getattr(args, "db", None)))
        return True, False
    if command == "/logs":
        run_logs(argparse.Namespace(n=5, failed=False, json=False))
        return True, False
    if command == "/clear":
        return True, True
    print(f"未知命令：{command}（输入 /help 查看可用命令）")
    return True, False


def run_chat(args) -> int:
    if getattr(args, "list_sessions", False):
        return print_sessions()

    agent, engine = build_agent(args)
    name, session = _resolve_session(args)
    history = sessions.turns_to_history(session.get("turns") or []) if session else []

    print("数据分析 Agent 已就绪（输入问题，输入 q 退出；/help 看命令）")
    print(f"引擎：{engine}")
    print(f"当前数据模式：{agent.mode_label}")
    print(f"会话：{name}（保存在 {sessions.sessions_dir()}）")
    if history:
        print(f"已恢复 {len(history) // 2} 轮历史，可直接接着问")
    else:
        print("支持追问，例如先问「各区域销售额」，再问「只看华东」")

    if getattr(args, "db", None) and not agent.ctx.has_db:
        # build_agent 已经降级为单表；这里只提示，不中断会话
        print(f"[提示] 指定的库不存在：{args.db}，已退回单表模式", file=sys.stderr)

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return EXIT_OK

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            print("再见！")
            return EXIT_OK
        if raw.startswith("/"):
            keep_going, clear = _handle_slash(raw.lower(), args)
            if clear:
                name = sessions.new_id()
                history = []
                print(f"已清空上下文，新会话：{name}")
            if not keep_going:
                print("再见！")
                return EXIT_OK
            continue

        try:
            state = agent.run(raw, history)
        except Exception as e:   # 兜底，避免单次失败中断整场会话
            print(f"[错误] {type(e).__name__}: {e}")
            continue

        print(f"\n===== 问题：{raw} =====")
        _print_human(state, raw)

        insight = state.get("insight") or state.get("answer") or ""
        try:
            sessions.append_turn(name, raw, insight,
                                 route=state.get("route") or "",
                                 statement=state.get("statement") or "")
        except sessions.SessionError as e:
            print(f"[提示] 会话未保存：{e}", file=sys.stderr)

        history = (history + [
            {"role": "user", "content": raw},
            {"role": "assistant", "content": insight},
        ])[-8:]   # 最近 4 轮，控制 token


def run_schema(args) -> int:
    ctx = make_context(args)
    # 与 MCP 的 `get_schema` 共用同一份 payload，避免两处字段名各自演化
    payload = schema_payload(ctx)
    tables = payload["tables"]

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK

    print(f"数据源：{ctx.mode_label}")
    if ctx.has_db:
        print(f"数据库：{ctx.db_path}")
        for table in tables:
            print(f"\n表 {table['name']}（{table['rows']} 行）")
            for col in table["columns"]:
                label = f"  {col['label']}" if col["label"] else ""
                print(f"  · {col['name']}{label}")
        if payload["relationships"]:
            print("\n表关系：")
            for rel in payload["relationships"]:
                print(f"  · {rel}")
    else:
        print(f"宽表：{ctx.csv_name}（找不到数据库，已降级为 Text-to-Pandas）")

    if payload["terms"]:
        print("\n指标口径：")
        for name, definition in payload["terms"].items():
            print(f"  · {name} = {definition}")
    return EXIT_OK


def run_logs(args) -> int:
    trace = (getattr(args, "trace", "") or "").strip()
    if trace:
        return _show_trace(trace, as_json=getattr(args, "json", False))

    rows = read_runs(limit=args.n, failed_only=args.failed)
    if getattr(args, "json", False):
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK

    if not rows:
        print("还没有运行记录。跑一次问答后就会写在：")
        print(f"  {log_path()}")
        return EXIT_OK

    print(f"{'时间':<20}{'结果':<6}{'路由':<8}{'重试':>4}{'行数':>6}{'耗时':>8}"
          f"{'tokens(入/出)':>16}  {'追踪号':<22}  问题")
    for row in rows:
        # 线性路线的汇总行没有 token 字段，显示"—"而不是 0/0（后者会被当成"真的没花钱"）
        tokens = (f"{row.get('tokens_in', 0)}/{row.get('tokens_out', 0)}"
                  if "tokens_in" in row else "—")
        print(f"{str(row.get('ts', '')):<20}"
              f"{'成功' if row.get('ok') else '失败':<6}"
              f"{str(row.get('route') or '-'):<8}"
              f"{int(row.get('retries') or 0):>4}"
              f"{int(row.get('rows') or 0):>6}"
              f"{str(row.get('elapsed') or 0) + 's':>8}"
              f"{tokens:>16}  "
              f"{str(row.get('trace_id') or '-'):<22}  {str(row.get('question') or '')[:28]}")
    failed = sum(1 for row in rows if not row.get("ok"))
    print(f"\n共 {len(rows)} 条，其中失败 {failed} 条。完整日志：{log_path()}")
    print("看某一次的各阶段耗时：python -m app.cli logs --trace <追踪号>")
    return EXIT_OK


def _show_trace(trace_id: str, as_json: bool = False) -> int:
    """把一次问答的阶段事件按执行顺序铺开（P2-2）。

    `logs` 表看向的是"历史有哪几次、成没成"，这里回答的是"那一次卡在哪"：
    understand 慢说明改写/取值校验拖了；execute 慢说明 SQL 本身重；
    链里出现两次 generate/execute 就是走了重试环。
    """
    events = read_events(trace_id)
    if not events:
        print(f"没有找到追踪号 {trace_id} 的阶段记录。", file=sys.stderr)
        print("可能是号码抄错（支持前缀与尾号），或这次问答发生在追踪功能上线之前。",
              file=sys.stderr)
        print("看最近几次：python -m app.cli logs", file=sys.stderr)
        return EXIT_USAGE

    if as_json:
        print(json.dumps(events, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK

    print(f"追踪号 {trace_id}（{len(events)} 个阶段）\n")
    traces = {str(e.get("trace_id", "")) for e in events}
    if len(traces) > 1:
        # 子串匹配可能命中多条链，明说比让用户以为"这次问了这么多步"好
        print(f"注意：这个号码匹配到 {len(traces)} 条链，下面按执行顺序全部列出，"
              f"每行末尾标注所属追踪号。\n")
    print(f"{'阶段':<12}{'耗时':>8}  关键信息")
    total = 0
    for event in events:
        ms = int(event.get("ms") or 0)
        total += ms
        extra = " · ".join(
            f"{k}={v}" for k, v in event.items()
            if k not in ("ts", "trace_id", "stage", "ms")
        )
        if len(traces) > 1:
            extra = f"{extra} · 追踪号={event.get('trace_id')}" if extra else \
                    f"追踪号={event.get('trace_id')}"
        print(f"{str(event.get('stage', '')):<12}{str(ms) + 'ms':>8}  {extra or '—'}")
    print(f"\n阶段合计 {total}ms（不含排队与未包装的部分）")

    summary = next((r for r in read_runs(limit=0)
                    if r.get("trace_id") == trace_id), None)
    if summary:
        # 线性路线没挂 token 回调：宁可说"未统计"，也不打印 0/0 假装统计过
        if "llm_calls" in summary:
            usage = (f"{int(summary.get('llm_calls') or 0)} 次模型调用 · "
                     f"tokens {summary.get('tokens_in', 0)}/{summary.get('tokens_out', 0)}")
        else:
            usage = "token 未统计（线性路线）"
        engine = {"graph": "状态机", "linear": "线性编排"}.get(summary.get("engine") or "")
        print(f"汇总：{'成功' if summary.get('ok') else '失败'} · "
              f"{int(summary.get('rows') or 0)} 行 · {usage} · "
              f"{summary.get('elapsed')}s" + (f"（{engine}）" if engine else ""))
        if summary.get("error"):
            print(f"错误：{summary['error']}")
    return EXIT_OK


def run_teach(args) -> int:
    """把一条「问题 → 正确 SQL」存成 few-shot（P1-1 修正回流）。

    为什么要先跑一遍 SQL：lesson 会被当成"可信上下文"拼进 Prompt，
    一条跑不通的语句等于教模型写错语句。默认验证，`--force` 才放行。
    """
    question = resolve_question(args.question)
    if not question:
        print("请提供答错的那个问题，例如：\n"
              "  python -m app.cli teach \"各城市的销售额是多少？\" "
              "--sql \"SELECT ...\"", file=sys.stderr)
        return EXIT_USAGE

    ctx = make_context(args)
    try:
        lesson = add_lesson(question, args.sql, note=args.note or "",
                            ctx=ctx, verify=not args.force)
    except LessonError as e:
        print(f"[拒绝入库] {e}", file=sys.stderr)
        return EXIT_USAGE

    total = len(load_lessons(ctx))
    print(f"已存入 {lesson['id']}（{'该拒答示范' if lesson['kind'] == 'defer' else '正确 SQL'}）")
    print(f"  问题：{lesson['question']}")
    print(f"  输出：{lesson['output'].splitlines()[0][:80]}")
    print(f"\n当前库共 {total} 条修正记录：{lessons_path(ctx)}")
    print("下次问到相似问题时会自动作为 few-shot 注入；不想生效可用 `lessons --rm` 删除。")
    return EXIT_OK


def run_lessons(args) -> int:
    """列出 / 删除 / 试算命中：修正记录的可解释入口。"""
    ctx = make_context(args)

    if args.rm:
        if remove_lesson(args.rm, ctx):
            print(f"已删除 {args.rm}。")
            return EXIT_OK
        print(f"没有找到 id 为 {args.rm} 的记录；用 `python -m app.cli lessons` 查看全部。",
              file=sys.stderr)
        return EXIT_USAGE

    items = load_lessons(ctx)

    if args.match:
        picked = select_lessons(args.match, items, ctx.enums, ctx.terms_dict)
        if args.json:
            print(json.dumps({"question": args.match,
                              "matched": [{"id": i.get("id"), "score": score_lesson(
                                  args.match, i, ctx.enums, ctx.terms_dict)} for i in picked]},
                             ensure_ascii=False, indent=2))
            return EXIT_OK
        print(f"问题：{args.match}")
        if not picked:
            print("不会注入任何记录（没有足够相关的；无关示范会带偏模型，宁可不用）。")
            return EXIT_OK
        print(f"会注入 {len(picked)} 条（按相关度降序）：")
        for item in picked:
            score = score_lesson(args.match, item, ctx.enums, ctx.terms_dict)
            print(f"  · {item.get('id')}  相关度 {score:.2f}  {item.get('question', '')[:50]}")
        return EXIT_OK

    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return EXIT_OK

    if not items:
        print("还没有修正记录。\n"
              "发现答错时这样教它：\n"
              "  python -m app.cli teach \"那个问题\" --sql \"正确的 SQL\" -n \"为什么这么改\"\n"
              "如果这类问题该拒答：--sql \"DEFER: 缺少成本字段\"")
        return EXIT_OK

    print(f"修正记录（{len(items)} 条）· 库指纹 {db_fingerprint(ctx)}")
    print(f"文件：{lessons_path(ctx)}\n")
    for item in items:
        kind = "拒答" if item.get("kind") == "defer" else "SQL"
        print(f"· {item.get('id')}  [{kind}]  {item.get('question', '')}")
        first = (item.get("output") or "").splitlines()
        if first:
            print(f"    → {first[0][:90]}")
        if item.get("note"):
            print(f"    备注：{item['note']}")
        print(f"    {item.get('created', '')}"
              f"{'  更新于 ' + item['updated'] if item.get('updated') else ''}")
    print("\n删除：python -m app.cli lessons --rm L001"
          "   看某问题会命中哪几条：--match \"问题\"")
    return EXIT_OK


def run_catalog(args) -> int:
    """从一个陌生的库生成 catalog 草稿（描述自动生成）。

    分两段跑：代码先实测（类型/样例/枚举/关系，零 token），模型再补语义
    （中文名/说明/口径）。`--dry-run` 只跑第一段 —— 先花零成本看看自动能认出什么，
    再决定值不值得花那一次调用。
    """
    db_path = args.db
    profile = profile_schema(db_path)
    if not profile["tables"]:
        print("[错误] 这个库里没有可读的表。", file=sys.stderr)
        for note in profile["notes"]:
            print(f"  · {note}", file=sys.stderr)
        return EXIT_FAILED

    if getattr(args, "dry_run", False):
        if getattr(args, "json", False):
            print(json.dumps(profile, ensure_ascii=False, indent=2, default=str))
            return EXIT_OK
        print(f"数据源：{db_path}")
        print(f"自动识别：{summarize_profile(profile)}")
        for note in profile["notes"]:
            print(f"  · 注意：{note}")
        print("\n[--dry-run] 只做代码侧识别，未调用模型。认出来的事实如下：\n")
        print(format_profile(profile))
        print("\n去掉 --dry-run 再跑一次，让它补上中文名、说明与指标口径。")
        return EXIT_OK

    existing = None
    if getattr(args, "merge", None):
        existing = load_catalog(args.merge)
        if not existing:
            print(f"[错误] 找不到可合并的词典（或文件是空的）：{args.merge}", file=sys.stderr)
            return EXIT_USAGE

    llm_raw = ""
    try:
        llm_raw = generate_descriptions(profile)
    except CatalogError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as e:
        print(f"[错误] 调用模型失败：{type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_FAILED

    llm_draft = parse_draft(llm_raw)
    if llm_draft is None:
        print("[错误] 模型返回的不是可解析的 JSON，未生成草稿（没写文件）。", file=sys.stderr)
        print(f"  原始输出前 300 字：{llm_raw[:300]}", file=sys.stderr)
        return EXIT_FAILED

    issues = validate_draft(llm_draft, profile)
    errors = [i for i in issues if i.get("level") == "error"]
    warns = [i for i in issues if i.get("level") == "warn"]
    draft = build_draft(profile, llm_draft, existing)

    if getattr(args, "json", False):
        print(json.dumps({"draft": draft, "issues": issues,
                          "profile_summary": summarize_profile(profile)},
                         ensure_ascii=False, indent=2, default=str))
        return EXIT_OK

    print(f"数据源：{db_path}")
    print(f"自动识别：{summarize_profile(profile)}")
    for note in profile["notes"]:
        print(f"  · 注意：{note}")

    if errors:
        print(f"\n[已丢弃] 模型提到的 {len(errors)} 处表/列在库里不存在（不能让它们进词典）：")
        for i in errors[:5]:
            print(f"  · {i['message']}")
        if len(errors) > 5:
            print(f"  · ...（还有 {len(errors) - 5} 条）")
    missing = [i for i in warns if i.get("kind") == "missing_column"]
    if missing:
        print(f"\n[待补] {len(missing)} 列没有中文名，草稿里已标 TODO")
    dropped_terms = [i for i in warns if i.get("kind") == "bad_term"]
    for i in dropped_terms:
        print(f"  · 已忽略口径：{i['message']}")

    out_path = args.out or "catalog.draft.yaml"
    text = render_catalog_yaml(
        draft, source=os.path.abspath(db_path),
        generated_at=time.strftime("%Y-%m-%d %H:%M"),
        notes=[f"模型提到的 {i['message'][:30]} 已丢弃" for i in errors[:3]],
    )
    out_abs = os.path.abspath(out_path)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_abs, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print(f"\n草稿已写入：{out_abs}")
    for line in describe_changes(draft, existing):
        print(f"  {line}")
    todos = count_todos(draft)
    if todos:
        print(f"  还有 {todos} 处 TODO 待确认（在文件里搜 TODO）")

    print("\n下一步：")
    print("  1) 打开草稿，搜 TODO，把缺的中文名与说明补上；")
    print("  2) 覆盖到 app/schema/catalog.yaml 后生效。")
    print("     只想补空缺、不动已有内容：--merge app/schema/catalog.yaml")
    return EXIT_OK


def run_serve(args) -> int:
    """以 MCP server 形态启动（stdio）。

    **这里不能 print 任何东西**：stdio 传输下 stdout 归 JSON-RPC 协议所有，
    多一行提示就握手失败。需要提示时一律走 stderr。
    延迟导入也是刻意的 —— mcp 是可选依赖，不能让其他命令因为没装它而 import 失败。
    """
    try:
        from app.mcp_server import SERVER_NAME, serve
    except ModuleNotFoundError as e:
        print(f"[启动失败] {e}", file=sys.stderr)
        return EXIT_FAILED
    return serve(db=getattr(args, "db", "") or "",
                 name=getattr(args, "name", "") or SERVER_NAME)


def run_version() -> int:
    ctx = AgentContext()
    print(f"数据分析 Agent v{VERSION}")
    print(f"模型：{LLM_MODEL}")
    print(f"数据源：{ctx.mode_label}")
    print("命令：ask / chat / schema / logs / teach / lessons / catalog / serve / version"
          "（-h 查看用法）")
    return EXIT_OK


# ---------------------------------------------------------------- 入口
def main(argv: Optional[list] = None) -> int:
    argv = normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    # `--db` 写错时明确报错：静默退回单表模式会让人以为问的是另一个库
    db_file = getattr(args, "db", None)
    if db_file and not os.path.exists(os.path.expanduser(db_file)):
        print(f"[错误] 数据库文件不存在：{db_file}", file=sys.stderr)
        return EXIT_USAGE

    command = args.command
    if command in ("ask", "a"):
        return run_ask(args)
    if command in ("chat", "c"):
        return run_chat(args)
    if command in ("schema", "s"):
        return run_schema(args)
    if command in ("logs", "l"):
        return run_logs(args)
    if command in ("teach", "t"):
        return run_teach(args)
    if command in ("lessons", "ls"):
        return run_lessons(args)

    if command == "serve":
        return run_serve(args)
    if command in ("catalog", "cat"):
        return run_catalog(args)
    if command == "version":
        return run_version()
    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
