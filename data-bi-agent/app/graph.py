"""LangGraph 状态机（对应功能 F3.2 / S3.3，阶段 3 实现）。

流程：

    START → understand → generate → execute ─┬─ 有结果 ──→ answer → END
                     ↑                        └─ 无结果 ──→ reflect
                     └──────────────────────────────────────┘（换路或回喂错误后重来）

设计说明：
- **understand** 负责多轮改写 + 路由；**reflect** 负责计数与换路；
- 重试上限由 max_retries 控制，默认 2 次（最多 3 次尝试：SQL → pandas → pandas 修正）；
- 生成层与执行层通过 state 解耦，任一路径失败都能走同一套反思逻辑。

与线性编排（app/agent.py）的关系：两者共用 AgentContext 与全部节点实现，
状态机多了条件边与重试环；CLI 默认用状态机，`--linear` 可切回线性版对比。
"""
from typing import Optional

from langgraph.graph import END, START, StateGraph

from app.agent import to_frame
from app.context import AgentContext
from app.nodes.answer import generate_insight, local_insight
from app.nodes.execute import execute_code
from app.nodes.generate import generate_code, generate_sql
from app.nodes.reflect import DEFAULT_MAX_RETRIES, make_reflect_node, route_after_execute
from app.nodes.understand import make_understand_node
from app.state import AgentState
from app.tools.chart_tool import build_chart, save_chart
from app.tools.sql_tool import run_sql


def make_generate_node(ctx: AgentContext):
    """生成节点：按当前路由产出 SQL 或 pandas 代码，重试时带上上次错误。"""

    def node(state: AgentState) -> dict:
        question = state.get("rewritten_question") or state.get("question", "")
        route = state.get("route", "sql")
        previous_error = state.get("error") or None

        try:
            if route == "sql":
                statement = generate_sql(state.get("schema_info", ""), state.get("terms", ""),
                                         question, previous_error=previous_error)
            else:
                statement = generate_code(state.get("schema_info", ""), state.get("terms", ""),
                                          question, previous_error=previous_error)
        except Exception as e:
            return {"statement": "", "error": f"生成失败：{type(e).__name__}: {e}"}

        return {"statement": statement}

    return node


def make_execute_node(ctx: AgentContext):
    """执行节点：SQL 走只读库，pandas 走受限命名空间。"""

    def node(state: AgentState) -> dict:
        statement = state.get("statement")
        if not statement:
            return {"result": None, "error": state.get("error") or "未生成可执行语句"}

        if state.get("route") == "sql" and ctx.has_db:
            r = run_sql(statement, ctx.db_path)
            if r["ok"]:
                return {"result": r["df"], "error": None}
            return {"result": None, "error": f"SQL {r['stage']} 阶段：{r['error']}"}

        r = execute_code(statement, ctx.df)
        if r["ok"]:
            return {"result": to_frame(r["result"]), "error": None}
        return {"result": None, "error": f"代码 {r['stage']} 阶段：{r['error']}"}

    return node


def make_answer_node(ctx: AgentContext, with_chart: bool = True,
                     with_insight: bool = True):
    """回答节点：出图 + 生成洞察；彻底失败时给出可读的说明。"""

    def node(state: AgentState) -> dict:
        result = state.get("result")

        if result is None:
            errors = state.get("errors") or []
            detail = state.get("error") or (errors[-1] if errors else "")
            answer = "未能完成分析。"
            if detail:
                answer += f"最后一次错误：{detail}"
            if errors:
                answer += f"（共尝试 {len(errors)} 次）"
            return {"insight": "", "answer": answer, "chart_type": "none", "chart_path": ""}

        question = state.get("rewritten_question") or state.get("question", "")
        update = {}

        if with_chart:
            try:
                fig, chart_type = build_chart(result, question)
                update["chart_type"] = chart_type
                update["chart_path"] = save_chart(fig) if fig is not None else ""
            except Exception as e:
                update["chart_type"] = "none"
                update["chart_path"] = ""
                update["error"] = f"图表生成失败：{type(e).__name__}: {e}"

        insight = generate_insight(question, result) if with_insight else local_insight(result)
        update["insight"] = insight
        update["answer"] = insight
        return update

    return node


def build_graph(ctx: Optional[AgentContext] = None, with_chart: bool = True,
                with_insight: bool = True, max_retries: int = DEFAULT_MAX_RETRIES):
    """编译状态机，返回可直接 invoke 的 graph。"""
    ctx = ctx or AgentContext()

    builder = StateGraph(AgentState)
    builder.add_node("understand", make_understand_node(ctx))
    builder.add_node("generate", make_generate_node(ctx))
    builder.add_node("execute", make_execute_node(ctx))
    builder.add_node("reflect", make_reflect_node(ctx))
    builder.add_node("answer", make_answer_node(ctx, with_chart, with_insight))

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "generate")
    builder.add_edge("generate", "execute")
    builder.add_conditional_edges(
        "execute", route_after_execute, {"answer": "answer", "reflect": "reflect"})
    builder.add_edge("reflect", "generate")
    builder.add_edge("answer", END)

    return builder.compile()


class GraphAgent:
    """状态机版门面：保持与 DataAgent 相近的调用方式，便于 CLI 切换。"""

    def __init__(self, ctx: Optional[AgentContext] = None, with_chart: bool = True,
                 with_insight: bool = True, max_retries: int = DEFAULT_MAX_RETRIES):
        self.ctx = ctx or AgentContext()
        self.max_retries = max_retries
        self.graph = build_graph(self.ctx, with_chart=with_chart,
                                 with_insight=with_insight, max_retries=max_retries)

    @property
    def mode_label(self) -> str:
        return self.ctx.mode_label

    def run(self, question: str, history: Optional[list] = None) -> dict:
        """执行一次问答，返回最终 state。"""
        initial: AgentState = {
            "question": question,
            "history": history or [],
            "retry_count": 0,
            "max_retries": self.max_retries,
        }
        return self.graph.invoke(initial)
