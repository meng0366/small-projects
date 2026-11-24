"""命令行问答入口（对应步骤 S1.7 / S2.7 / S3.6，功能 F4.1 的命令行版）。

用法：
  交互模式：python -m app.cli
  单次提问：python -m app.cli "各区域的销售额是多少？"

可选参数：
  --linear       使用线性编排（无重试、无多轮记忆），默认用 LangGraph 状态机
  --no-chart     不生成图表
  --no-insight   不生成洞察文案（省一次模型调用）
  --quiet        不打印生成的 SQL / 代码
  --pandas       仅线性模式可用：强制走单表 pandas

交互模式下支持追问（多轮记忆），例如先问"各区域销售额"，再问"只看华东"。
"""
import sys

from app.agent import DataAgent
from app.graph import GraphAgent
from app.nodes.execute import format_result

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文打印报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _print_result(df, flags: set, chart_path: str, chart_type: str, insight: str) -> None:
    """打印结果、图表与洞察（两条编排路径共用）。"""
    print(f"\n[分析结果]（{len(df)} 行）")
    print(format_result(df))

    if chart_path:
        from app.tools.chart_tool import CHART_TYPE_LABEL
        print(f"\n[图表] {CHART_TYPE_LABEL.get(chart_type, chart_type)} → {chart_path}")
    elif chart_type == "none" and "--no-chart" not in flags:
        print("\n[图表] 结果不适合可视化，已跳过")

    if insight:
        print(f"\n[洞察] {insight}")


def _route_label(route: str) -> str:
    return "SQLite 多表（Text-to-SQL）" if route == "sql" else "单表宽表（Text-to-Pandas）"


def render_graph(agent: GraphAgent, question: str, history: list, flags: set) -> dict:
    """状态机路径：执行并打印全过程，返回 state（供多轮记忆使用）。"""
    print(f"\n===== 问题：{question} =====")
    state = agent.run(question, history)

    rewritten = state.get("rewritten_question") or question
    if rewritten != question:
        print(f"[改写] {rewritten}")

    print(f"\n[路由] {_route_label(state.get('route', ''))}")

    retry_count = state.get("retry_count", 0)
    if retry_count:
        print(f"[重试] 共重试 {retry_count} 次后完成")
        for err in state.get("errors") or []:
            print(f"  · {err}")

    statement = state.get("statement")
    if statement and "--quiet" not in flags:
        print(f"\n[生成的{'SQL' if state.get('route') == 'sql' else '代码'}]")
        print(statement)

    df = state.get("result")
    if df is None:
        print(f"\n{state.get('answer') or '[未能回答]'}")
        return state

    _print_result(df, flags, state.get("chart_path", ""),
                  state.get("chart_type", "none"), state.get("insight", ""))
    return state


def render_linear(agent: DataAgent, question: str, flags: set) -> None:
    """线性路径（无重试、无记忆）。"""
    print(f"\n===== 问题：{question} =====")
    res = agent.ask(
        question,
        with_chart="--no-chart" not in flags,
        with_insight="--no-insight" not in flags,
        force_pandas="--pandas" in flags,
    )

    if not res.ok:
        print("\n[未能回答]")
        for err in res.errors:
            print(f"  · {err}")
        return

    print(f"\n[路由] {_route_label(res.route)}")
    if "--quiet" not in flags:
        print(f"\n[生成的{'SQL' if res.route == 'sql' else '代码'}]")
        print(res.statement)

    _print_result(res.df, flags, res.chart_path, res.chart_type, res.insight)


def main() -> None:
    argv = sys.argv[1:]
    flags = {a for a in argv if a.startswith("--")}
    question = " ".join(a for a in argv if not a.startswith("--")).strip()

    use_graph = "--linear" not in flags
    with_chart = "--no-chart" not in flags
    with_insight = "--no-insight" not in flags

    if use_graph:
        agent = GraphAgent(with_chart=with_chart, with_insight=with_insight)
        engine = "LangGraph 状态机（含重试与多轮记忆）"
    else:
        agent = DataAgent()
        engine = "线性编排（无重试、无记忆）"

    if question:
        if use_graph:
            render_graph(agent, question, [], flags)
        else:
            render_linear(agent, question, flags)
        return

    print("数据分析 Agent 已就绪（输入问题，输入 q 退出）")
    print(f"引擎：{engine}")
    print(f"当前数据模式：{agent.mode_label}")
    if use_graph:
        print("支持追问，例如先问「各区域销售额」，再问「只看华东」")

    history: list = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if q.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if not q:
            continue
        try:
            if use_graph:
                state = render_graph(agent, q, history, flags)
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant",
                                "content": state.get("insight") or state.get("answer") or ""})
                history = history[-8:]   # 只保留最近 4 轮，控制 token
            else:
                render_linear(agent, q, flags)
        except Exception as e:  # 兜底，避免单次失败中断会话
            print(f"[错误] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
