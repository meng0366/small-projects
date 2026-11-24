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
    is_aggregate_empty,
    is_empty_result,
    make_reflect_node,
    route_after_execute,
)
from app.nodes.understand import make_understand_node
from app.state import AgentState
from app.tools.chart_tool import build_chart, save_chart
from app.tools.enum_guard import build_clarify_answer
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
        # 记下每轮语句：answer 节点据此判断"重试时到底有没有改过条件"
        statements = list(state.get("statements") or [])
        if statement:
            statements.append(statement)
        history = {"statements": statements}

        if not statement:
            return {"result": None, "error": state.get("error") or "未生成可执行语句",
                    **history}

        if state.get("route") == "sql" and ctx.has_db:
            r = run_sql(statement, ctx.db_path)
            if r["ok"]:
                empty = is_empty_result(r["df"])
                return {"result": r["df"], "error": None, "empty_result": empty, **history}
            return {"result": None, "error": f"SQL {r['stage']} 阶段：{r['error']}",
                    "empty_result": False, **history}

        r = execute_code(statement, ctx.df)
        if r["ok"]:
            frame = to_frame(r["result"])
            empty = is_empty_result(frame)
            return {"result": frame, "error": None, "empty_result": empty, **history}
        return {"result": None, "error": f"代码 {r['stage']} 阶段：{r['error']}",
                "empty_result": False, **history}

    return node


def build_empty_answer(ctx: AgentContext, state: AgentState, aggregate: bool) -> str:
    """空结果的如实说明：不编原因，摆事实（库里到底覆盖了什么）。

    真实试跑暴露过两个问题，这段文案是针对性的：
    - 聚合空（SUM 返回 NULL）曾被 LLM 说成"数据缺失或需核查数据源"，把用户引向错误方向；
    - 只给"可能原因"（取值不一致/时间超范围）不够，用户还得自己猜库里有啥。
    所以直接把实测取值与时间范围摆出来，让用户一眼看出该换成什么条件。
    """
    retried = bool(state.get("empty_retried"))
    statements = state.get("statements") or []
    unchanged = len(statements) >= 2 and statements[-1] == statements[-2]

    parts = ["查询已执行成功，但当前条件下没有匹配到数据。"]
    if retried and unchanged:
        parts.append("系统自动复核了一次：重试的查询条件与首次完全一致，"
                     "说明条件本身没写错，是这个范围确实没有数据。")
    elif retried:
        parts.append("系统已自动放宽条件重查一次，结果仍为空。")

    profile = ctx.data_profile_text
    if profile:
        parts.append("")
        parts.append("库里实际覆盖的范围：")
        parts.append(profile)

    parts.append("")
    parts.append("可以按上面列出的取值或时间范围换个条件再问；"
                 "如果确认要的就是这个范围，那就是没有数据。")
    return "\n".join(parts)


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

        # 空结果：执行成功但没数据。分两种表现，处置相同、说明不同：
        # - 0 行（明细型，已自动放宽重试过一次）；
        # - 1 行 NULL（聚合型，如 SUM 无匹配，按策略不重试）。
        # 两种情况都不让模型"编洞察" —— 空结果没什么可洞察的，还容易胡编。
        if is_empty_result(result) or is_aggregate_empty(result):
            answer = build_empty_answer(ctx, state, aggregate=is_aggregate_empty(result))
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


def make_clarify_node(ctx: AgentContext):
    """澄清节点：用户问的取值库里没有时，直接说明并给出可选值（P0-1b）。

    不再走生成：模型知道取值列表，但遇到对不上的词会自己挑个"最近的"
    （"办公用品"→"配件"），产出的数字口径是错的，比不回答更糟。
    """

    def node(state: AgentState) -> dict:
        answer = build_clarify_answer(state.get("unknown_enum") or {})
        return {"answer": answer, "insight": answer,
                "chart_type": "none", "chart_path": "", "result": None}

    return node


def route_after_understand(state: AgentState) -> str:
    """条件边：取值对不上先澄清，对得上才去生成。"""
    return "clarify" if state.get("unknown_enum") else "generate"


def build_graph(ctx: Optional[AgentContext] = None, with_chart: bool = True,
                with_insight: bool = True, max_retries: int = DEFAULT_MAX_RETRIES):
    """编译状态机，返回可直接 invoke 的 graph。"""
    ctx = ctx or AgentContext()

    builder = StateGraph(AgentState)
    builder.add_node("understand", make_understand_node(ctx))
    builder.add_node("clarify", make_clarify_node(ctx))
    builder.add_node("generate", make_generate_node(ctx))
    builder.add_node("execute", make_execute_node(ctx))
    builder.add_node("reflect", make_reflect_node(ctx))
    builder.add_node("answer", make_answer_node(ctx, with_chart, with_insight))

    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand", route_after_understand,
        {"generate": "generate", "clarify": "clarify"})
    builder.add_edge("clarify", END)
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
        unknown = state.get("unknown_enum")

        # 追问建议：纯规则计算（零 token），失败也不该影响本次回答
        try:
            if unknown:
                # 澄清场景下通用建议没意义，直接把可选值做成能点的问题
                state["suggestions"] = [
                    f"{v}的销售额是多少？"
                    for v in (unknown.get("values") or [])[:3]]
            else:
                state["suggestions"] = suggest_followups(
                    state.get("rewritten_question") or raw_question, result)
        except Exception:
            state["suggestions"] = []

        log_run(
            ok=result is not None or bool(unknown),
            clarify=bool(unknown),
            route=state.get("route", ""),
            retries=int(state.get("retry_count") or 0),
            rows=0 if result is None else int(len(result)),
            empty=is_empty_result(result) or is_aggregate_empty(result),
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
