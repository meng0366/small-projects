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

import time

from langgraph.graph import END, START, StateGraph

from app.agent import to_frame
from app.config import MAX_QUESTION_CHARS
from app.context import AgentContext
from app.logging_utils import TokenCounter, log_run
from app.nodes.answer import generate_insight, local_insight
from app.nodes.execute import execute_code
from app.nodes.generate import generate_code, generate_sql
from app.nodes.reflect import (
    EMPTY_RESULT_NOTE,
    DEFAULT_MAX_RETRIES,
    is_empty_result,
    make_reflect_node,
    route_after_execute,
)
from app.nodes.understand import make_understand_node
from app.state import AgentState
from app.tools.chart_tool import build_chart, save_chart
from app.tools.sql_tool import run_sql
from app.tools.suggest_tool import suggest_followups


def make_generate_node(ctx: AgentContext):
    """生成节点：按当前路由产出 SQL 或 pandas 代码，重试时带上上次错误。"""

    def node(state: AgentState) -> dict:
        question = state.get("rewritten_question") or state.get("question", "")
        route = state.get("route", "sql")
        previous_error = state.get("error") or None
        # 空结果与报错要分开提示：空结果让模型"放宽条件"，报错让模型"修语法"。
        # 用「上一轮执行结果是否为空」判断，而不是「是否重试过」——
        # 空结果之后再报错时，必须切回报错模板。
        previous_empty = bool(state.get("empty_result"))

        try:
            if route == "sql":
                statement = generate_sql(state.get("schema_info", ""), state.get("terms", ""),
                                         question, previous_error=previous_error,
                                         previous_empty=previous_empty)
            else:
                statement = generate_code(state.get("schema_info", ""), state.get("terms", ""),
                                          question, previous_error=previous_error,
                                          previous_empty=previous_empty)
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
                empty = is_empty_result(r["df"])
                return {"result": r["df"], "error": None, "empty_result": empty}
            return {"result": None, "error": f"SQL {r['stage']} 阶段：{r['error']}",
                    "empty_result": False}

        r = execute_code(statement, ctx.df)
        if r["ok"]:
            frame = to_frame(r["result"])
            empty = is_empty_result(frame)
            return {"result": frame, "error": None, "empty_result": empty}
        return {"result": None, "error": f"代码 {r['stage']} 阶段：{r['error']}",
                "empty_result": False}

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

        # 空结果：执行成功但没数据。已经自动重试过一次（放宽条件）仍为空，
        # 就如实说明，不再让模型"编"一段洞察 —— 空表没什么可洞察的，还容易胡编。
        if is_empty_result(result):
            retried = bool(state.get("empty_retried"))
            answer = "查询已执行成功，但当前条件下没有匹配到数据。"
            answer += ("已自动尝试放宽条件重查一次，仍为空；"
                       if retried else "")
            answer += ("可能原因：筛选条件过严或条件叠加后无交集、"
                       "地区/品类/客户等级等取值与库中写法不一致、"
                       "或所问时间段超出数据覆盖范围（本库为 2025 年数据）。"
                       "可以换个时间段或减少筛选条件再问。")
            return {"insight": answer, "answer": answer,
                    "chart_type": "none", "chart_path": ""}

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

    def run(self, question: str, history: Optional[list] = None,
            config: Optional[dict] = None) -> dict:
        """执行一次问答，返回最终 state。

        config 会原样透传给 LangGraph，可用于挂回调（评测时统计 token 用量）。
        超长提问会被截断以保护上下文，截断情况记进日志。
        """
        started = time.perf_counter()
        raw_question = (question or "").strip()
        truncated = len(raw_question) > MAX_QUESTION_CHARS
        if truncated:
            raw_question = raw_question[:MAX_QUESTION_CHARS]

        initial: AgentState = {
            "question": raw_question,
            "history": history or [],
            "retry_count": 0,
            "max_retries": self.max_retries,
        }

        # 自动挂一个 token 计数器：成本监控写进日志，调用方也可再传自己的回调
        counter = TokenCounter()
        merged_config = dict(config or {})
        callbacks = merged_config.get("callbacks") or []
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        merged_config["callbacks"] = [*callbacks, counter]

        try:
            state = self.graph.invoke(initial, config=merged_config)
        except Exception as e:
            log_run(ok=False, route="", retries=0, rows=0, chart="",
                    elapsed=round(time.perf_counter() - started, 2),
                    question=raw_question[:200], question_len=len(raw_question),
                    truncated=truncated, error=f"{type(e).__name__}: {e}",
                    llm_calls=counter.calls, tokens_in=counter.input_tokens,
                    tokens_out=counter.output_tokens)
            raise

        result = state.get("result")

        # 追问建议：纯规则计算（零 token），失败也不该影响本次回答
        try:
            state["suggestions"] = suggest_followups(
                state.get("rewritten_question") or raw_question, result)
        except Exception:
            state["suggestions"] = []

        log_run(
            ok=result is not None,
            route=state.get("route", ""),
            retries=int(state.get("retry_count") or 0),
            rows=0 if result is None else int(len(result)),
            empty=is_empty_result(result),
            chart=state.get("chart_type", ""),
            elapsed=round(time.perf_counter() - started, 2),
            question=raw_question[:200],
            question_len=len(raw_question),
            truncated=truncated,
            error=(state.get("error") or "")[:300],
            errors=len(state.get("errors") or []),
            llm_calls=counter.calls,
            tokens_in=counter.input_tokens,
            tokens_out=counter.output_tokens,
        )
        return state
