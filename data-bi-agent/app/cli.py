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
import sqlite3
import sys
import time
from typing import Optional

from app import sessions
from app.agent import DataAgent
from app.config import LLM_MODEL
from app.context import AgentContext
from app.graph import GraphAgent
from app.logging_utils import log_path, read_events, read_runs
from app.render import (
    SUPPORTED_FORMATS,
    UnsupportedFormat,
    export,
    normalize_format,
    render,
)
from app.schema.loader import list_tables, load_catalog
from app.tools.chart_tool import CHART_TYPE_LABEL
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
from app.tools.suggest_tool import suggest_followups

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文打印报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

VERSION = "0.8.0"
"""与项目阶段对齐：阶段 6 是计划外增强（CLI 子命令、会话续聊、追问建议）；
0.7 是改进清单 P1-1（修正回流 teach/lessons）；0.8 是 P1-2（指标归因）。"""

SUBCOMMANDS = ("ask", "a", "chat", "c", "schema", "s", "logs", "l",
               "teach", "t", "lessons", "ls", "version", "help")
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
        ),
    )
    sub = parser.add_subparsers(dest="command")

    def add_engine_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", metavar="PATH", help="使用指定的 SQLite 库文件（默认 data/sales.db）")
        p.add_argument("--no-chart", action="store_true", help="不生成图表")
        p.add_argument("--no-insight", action="store_true", help="不生成洞察文案（省一次模型调用）")
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
    return AgentContext(db_file=db_file) if db_file else AgentContext()


def build_agent(args):
    """按参数构建 Agent，返回 (agent, 引擎说明)。"""
    ctx = make_context(args)
    with_chart = not getattr(args, "no_chart", False)
    with_insight = not getattr(args, "no_insight", False)
    if getattr(args, "linear", False):
        return DataAgent(ctx=ctx), "线性编排（无重试、无记忆）"
    return GraphAgent(ctx, with_chart=with_chart, with_insight=with_insight), \
        "LangGraph 状态机（含重试与多轮记忆）"


def resolve_question(question: Optional[str]) -> str:
    """取问题：支持 `-` 或空参数时从标准输入读（管道友好）。"""
    if question is None or question == "-":
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
        return ""
    return question.strip()


def _state_from_linear(res, question: str) -> dict:
    """把线性编排的结果整理成与状态机一致的 dict，后面统一渲染。"""
    return {
        "route": res.route,
        "statement": res.statement,
        "result": res.df,
        "insight": res.insight,
        "chart_path": res.chart_path,
        "chart_type": res.chart_type,
        "errors": list(res.errors),
        "retry_count": 0,
        "answer": "" if res.ok else "未能完成分析。",
        "suggestions": suggest_followups(question, res.df),
    }


def ask_once(args, question: str) -> dict:
    """执行一次问答，返回状态 dict（附 `_elapsed`）。"""
    started = time.perf_counter()
    agent, _ = build_agent(args)

    if getattr(args, "linear", False):
        res = agent.ask(
            question,
            with_chart=not args.no_chart,
            with_insight=not args.no_insight,
            force_pandas=args.pandas,
        )
        state = _state_from_linear(res, question)
    else:
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
    """`-f json` 的结构化输出：脚本里最需要的是语句、数据、指标，而不是排版。"""
    df = state.get("result")
    att = state.get("attribution") or {}
    attribution = None
    if att.get("ok"):
        attribution = {
            "compare": {"base": att.get("base_label"), "current": att.get("current_label")},
            "dimension": att.get("dim"),
            "metric": att.get("metric"),
            "base_total": att.get("base_total"),
            "current_total": att.get("current_total"),
            "delta": att.get("delta"),
            "pct": att.get("pct"),
            "additive": att.get("additive"),
            "rows": json.loads(render(att["table"], "json")),
            "notes": att.get("notes") or [],
        }
    return {
        "question": question,
        "rewritten": state.get("rewritten_question") or "",
        "trace_id": state.get("trace_id") or "",
        "ok": df is not None,
        "route": state.get("route") or "",
        "statement": state.get("statement") or "",
        "columns": [] if df is None else [str(c) for c in df.columns],
        "rows": 0 if df is None else int(len(df)),
        "data": [] if df is None else json.loads(render(df, "json")),
        "attribution": attribution,
        "insight": state.get("insight") or "",
        "suggestions": state.get("suggestions") or [],
        "retries": int(state.get("retry_count") or 0),
        "errors": state.get("errors") or [],
        "elapsed": state.get("_elapsed"),
    }


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


def _db_columns(db_path: str) -> dict:
    """读取每张表的字段名与行数（只读连接，仅用于展示）。"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        result = {}
        for name in list_tables(db_path):
            rows = con.execute(f'PRAGMA table_info("{name}")').fetchall()
            count = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            result[name] = {"columns": [r[1] for r in rows], "rows": count}
        return result
    finally:
        con.close()


def run_schema(args) -> int:
    ctx = make_context(args)
    catalog = load_catalog()
    labels = {}
    for table in catalog.get("tables") or []:
        for column, meta in (table.get("columns") or {}).items():
            if isinstance(meta, dict) and meta.get("label"):
                labels[f"{table.get('name')}.{column}"] = meta["label"]

    tables = []
    if ctx.has_db:
        info = _db_columns(ctx.db_path)
        for name, meta in info.items():
            tables.append({
                "name": name,
                "rows": meta["rows"],
                "columns": [
                    {"name": col, "label": labels.get(f"{name}.{col}", "")}
                    for col in meta["columns"]
                ],
            })

    payload = {
        "mode": ctx.mode_label,
        "db": ctx.db_path,
        "tables": tables,
        "terms": catalog.get("terms") or {},
        "relationships": catalog.get("relationships") or [],
    }

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
        tokens = f"{row.get('tokens_in', 0)}/{row.get('tokens_out', 0)}"
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
        print("只有状态机路线（默认）会产生分阶段事件；`--linear` 不经过节点，没有链。",
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
        print(f"汇总：{'成功' if summary.get('ok') else '失败'} · "
              f"{int(summary.get('rows') or 0)} 行 · "
              f"{int(summary.get('llm_calls') or 0)} 次模型调用 · "
              f"tokens {summary.get('tokens_in', 0)}/{summary.get('tokens_out', 0)} · "
              f"{summary.get('elapsed')}s")
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


def run_version() -> int:
    ctx = AgentContext()
    print(f"数据分析 Agent v{VERSION}")
    print(f"模型：{LLM_MODEL}")
    print(f"数据源：{ctx.mode_label}")
    print("命令：ask / chat / schema / logs / teach / lessons / version（-h 查看用法）")
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
    if command == "version":
        return run_version()
    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
