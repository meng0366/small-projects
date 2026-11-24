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

追踪（P2-2）：每个节点经 `_traced()` 包装，执行完写一行阶段事件；
同一次问答的所有事件与末尾的汇总行共用一个 `trace_id`，`cli logs --trace <id>` 可还原链路。
包装器不改节点代码，节点以后增删都不用想着"记得补日志"。
"""
from typing import Optional

import time

from langgraph.graph import END, START, StateGraph

from app.agent import to_frame
from app.config import MAX_QUESTION_CHARS
from app.context import AgentContext
from app.logging_utils import TokenCounter, log_event, log_run, new_trace_id
from app.nodes.answer import generate_insight, local_insight
from app.nodes.execute import execute_code
from app.nodes.generate import generate_attribution_sql, generate_code, generate_sql
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
from app.tools.attribution import attribution_chart_frame, decompose
from app.tools.enum_guard import build_clarify_answer
from app.tools.lesson_store import select_lessons
from app.tools.scope_guard import check_scope, looks_like_sql, parse_defer
from app.tools.sql_tool import run_sql
from app.tools.suggest_tool import suggest_followups


def make_generate_node(ctx: AgentContext):
    """生成节点：按当前路由产出 SQL 或 pandas 代码，重试时带上上次错误。

    生成前会挑出**与本问题相关**的修正记录（P1-1）作为 few-shot：
    用户 teach 的"这类问题该怎么写 / 这类问题该拒答"，比通用示例更贴近当前问题。
    挑不到相关的就一条都不注入——无关的示范会把模型带偏。
    """

    def node(state: AgentState) -> dict:
        question = state.get("rewritten_question") or state.get("question", "")
        route = state.get("route", "sql")
        previous_error = state.get("error") or None
        # 空结果与报错要分开提示：空结果让模型"放宽条件"，报错让模型"修语法"。
        # 用「上一轮执行结果是否为空」判断，而不是「是否重试过」——
        # 空结果之后再报错时，必须切回报错模板。
        previous_empty = bool(state.get("empty_result"))

        lessons = []
        if ctx.use_lessons:
            try:
                lessons = select_lessons(question, ctx.lessons, ctx.enums, ctx.terms_dict)
            except Exception:
                lessons = []      # 沉淀层出问题不能影响问答本身
        used = [item.get("id", "") for item in lessons]

        intent = state.get("attribution_intent")
        try:
            if intent and route == "sql":
                # 归因问题要的不是"一个总数"，而是能拆的交叉表（P1-2）
                statement = generate_attribution_sql(
                    state.get("schema_info", ""), state.get("terms", ""), question, intent,
                    previous_error=previous_error, previous_empty=previous_empty,
                    lessons=lessons)
            elif route == "sql":
                statement = generate_sql(state.get("schema_info", ""), state.get("terms", ""),
                                         question, previous_error=previous_error,
                                         previous_empty=previous_empty, lessons=lessons)
            else:
                statement = generate_code(state.get("schema_info", ""), state.get("terms", ""),
                                          question, previous_error=previous_error,
                                          previous_empty=previous_empty, lessons=lessons)
        except Exception as e:
            return {"statement": "", "error": f"生成失败：{type(e).__name__}: {e}",
                    "lessons_used": used}

        return {"statement": statement, "lessons_used": used}

    return node


def make_execute_node(ctx: AgentContext):
    """执行节点：SQL 走只读库，pandas 走受限命名空间。

    执行前有两道守卫（P0-2b，都由真实评测暴露出来）：
    1. **模型说答不了**（`DEFER: 原因` 或整段自然语言）时不要当 SQL 执行 ——
       以前这种句子会报语法错，重试后就跑到另一个问题上去，最后把别的问题的答案交出去；
    2. **空结果重试不得改动用户的条件** —— 实测会把「2024 年 3 月」改成「2025-03」
       拿回 100 行真实数据冒充，用户完全看不出来。
    """

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

        # 守卫 1：模型明确说"库里没有这个字段/数据"
        reason = parse_defer(statement)
        if reason is None and state.get("route") == "sql" and not looks_like_sql(statement):
            # 整段自然语言（如"数据库中不存在退货相关字段"）也当作拒答，别拿去执行
            reason = statement.strip()[:200] or "模型判断现有数据无法回答这个问题"
        if reason:
            return {"result": None, "defer": reason, "error": None, **history}

        # 守卫 2 的前半段：先记下这轮是不是"空结果之后的重试"
        retrying_after_empty = bool(state.get("empty_result"))
        question = state.get("rewritten_question") or state.get("question", "")

        if state.get("route") == "sql" and ctx.has_db:
            r = run_sql(statement, ctx.db_path)
            if r["ok"]:
                frame = r["df"]
                problems = check_scope(question, statement, ctx.enums) \
                    if retrying_after_empty and not is_empty_result(frame) else []
                if problems:
                    # 拿到了数据，但是靠改动用户的条件换来的（如 2024→2025）→ 丢掉，不许用
                    return {"result": None, "defer": "；".join(problems),
                            "scope_violation": problems, **history}
                return {"result": frame, "error": None,
                        "empty_result": is_empty_result(frame), **history}
            return {"result": None, "error": f"SQL {r['stage']} 阶段：{r['error']}",
                    "empty_result": False, **history}

        r = execute_code(statement, ctx.df)
        if r["ok"]:
            frame = to_frame(r["result"])
            problems = check_scope(question, statement, ctx.enums) \
                if retrying_after_empty and not is_empty_result(frame) else []
            if problems:
                return {"result": None, "defer": "；".join(problems),
                        "scope_violation": problems, **history}
            return {"result": frame, "error": None,
                    "empty_result": is_empty_result(frame), **history}
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


def build_defer_answer(ctx: AgentContext, state: AgentState) -> str:
    """拒答说明：说清"为什么答不了"，而不是给一个换了口径的数字。

    两种情况共用这一段：
    - 模型声明缺字段（毛利率要成本、退货率要退货表）；
    - 空结果重试时把用户的条件改了（2024 年改成 2025 年）—— 这种最危险，
      数字看起来完全正常，用户不会怀疑，所以必须在这里挑明并拒绝。
    """
    reason = (state.get("defer") or "").strip()
    problems = state.get("scope_violation") or []
    parts = []

    if problems:
        parts.append("没法按你给的条件出数，也没有拿别的范围顶替。")
        parts.append("")
        parts.append("自动复核时发现，重试的查询把你限定的条件改动过：")
        parts.extend(f"- {item}" for item in problems)
        parts.append("")
        parts.append("你的条件本身没问题，是这个范围在库里没有数据 —— "
                     "改条件之后算出来的数字不能当成这个问题的答案，所以没有给。")
    else:
        parts.append("这个问题用现有数据答不了，也不该编一个数字给你。")
        if reason:
            parts.append("")
            parts.append(f"原因：{reason}")

    profile = ctx.data_profile_text
    if profile:
        parts.append("")
        parts.append("库里实际覆盖的范围：")
        parts.append(profile)

    parts.append("")
    parts.append("如果要的是库里有的口径，把条件改成上面列出的范围或取值，我再算。")
    return "\n".join(parts)


def _chart_for(frame, question: str) -> dict:
    """出图并落盘（图表是锦上添花，失败不该影响结论）。"""
    try:
        fig, chart_type = build_chart(frame, question)
        return {"chart_type": chart_type,
                "chart_path": save_chart(fig) if fig is not None else ""}
    except Exception as e:
        return {"chart_type": "none", "chart_path": "",
                "error": f"图表生成失败：{type(e).__name__}: {e}"}


def make_answer_node(ctx: AgentContext, with_chart: bool = True,
                     with_insight: bool = True):
    """回答节点：出图 + 生成洞察；彻底失败时给出可读的说明。"""

    def node(state: AgentState) -> dict:
        result = state.get("result")

        # 拒答：模型声明缺字段，或重试时改动了用户的条件（P0-2b）。
        # 这时一个数字都不能给 —— 给了就是把别的问题的答案冒名顶替。
        if state.get("defer"):
            answer = build_defer_answer(ctx, state)
            return {"insight": answer, "answer": answer,
                    "chart_type": "none", "chart_path": ""}

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

        # 归因（P1-2）：拆得开就直接给"变化来自哪里"，数字全由代码算。
        # 拆不开不硬凑 —— 说明原因，退回普通明细，用户至少拿到的还是真实数据。
        intent = state.get("attribution_intent")
        note = ""
        if intent:
            att = decompose(result, intent, question)
            update["attribution"] = att
            if att.get("ok"):
                update["insight"] = att["summary"]
                update["answer"] = att["summary"]
                if with_chart:
                    update.update(_chart_for(attribution_chart_frame(att), question))
                return update
            note = (f"\n\n（没能拆出归因：{att.get('reason')}。上面是查询结果本身；"
                    f"如果想按某个维度拆，可以在问题里说明，例如「按品类拆」。）")

        if with_chart:
            update.update(_chart_for(result, question))

        insight = generate_insight(question, result) if with_insight else local_insight(result)
        update["insight"] = insight + note
        update["answer"] = insight + note
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


# ---------------------------------------------------------------- 链路追踪（P2-2）
#
# 汇总行只说"最后怎么样"，看不出"慢在哪一步、重试发生在哪、错在哪一层"。
# 这里给每个节点包一层：执行完补一行阶段事件，全部共用同一个 trace_id。
# 用白名单挑字段 —— 绝不能把结果表或 SQL 全文写进日志（日志会被翻、会被分享）。

_TRACE_FIELDS = (
    # (状态里的键, 日志字段名, 取值转换)
    ("route", "route", str),
    ("retry_count", "retry", int),
    ("statement", "statement_len", lambda v: len(str(v or ""))),
    ("result", "rows", lambda v: 0 if v is None else len(v)),
    ("error", "error", lambda v: str(v or "")[:200]),
    ("unknown_enum", "clarify", bool),
    ("defer", "defer", bool),
    ("scope_violation", "scope_violation", bool),
    ("attribution", "attribution", lambda v: bool((v or {}).get("ok"))),
    ("chart_type", "chart", str),
    ("lessons_used", "lessons", len),
    ("errors", "errors", len),
)


def _trace_fields(update: dict) -> dict:
    """只挑本次节点真正写过的键；0 / False / 空串这类"没发生"的值不写。"""
    fields = {}
    for key, name, cast in _TRACE_FIELDS:
        if key not in update:
            continue
        try:
            value = cast(update.get(key))
        except Exception:
            continue
        if value in (0, False, "", None) and name != "rows":
            continue          # rows=0 是有意义的（查不到数据），其余零值只是噪音
        fields[name] = value
    return fields


def _traced(name: str, node):
    """节点包装器：记耗时与关键字段；节点抛异常也要留痕，否则链会缺一段。"""

    def wrapped(state: AgentState) -> dict:
        started = time.perf_counter()
        trace_id = (state or {}).get("trace_id", "")
        try:
            update = node(state) or {}
        except Exception as e:
            log_event(trace_id, name,
                      ms=round((time.perf_counter() - started) * 1000),
                      failed=True, error=f"{type(e).__name__}: {e}"[:200])
            raise
        log_event(trace_id, name,
                  ms=round((time.perf_counter() - started) * 1000),
                  **_trace_fields(update))
        return update

    return wrapped


def build_graph(ctx: Optional[AgentContext] = None, with_chart: bool = True,
                with_insight: bool = True, max_retries: int = DEFAULT_MAX_RETRIES):
    """编译状态机，返回可直接 invoke 的 graph。"""
    ctx = ctx or AgentContext()

    builder = StateGraph(AgentState)
    builder.add_node("understand", _traced("understand", make_understand_node(ctx)))
    builder.add_node("clarify", _traced("clarify", make_clarify_node(ctx)))
    builder.add_node("generate", _traced("generate", make_generate_node(ctx)))
    builder.add_node("execute", _traced("execute", make_execute_node(ctx)))
    builder.add_node("reflect", _traced("reflect", make_reflect_node(ctx)))
    builder.add_node("answer", _traced("answer",
                                      make_answer_node(ctx, with_chart, with_insight)))

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

        # 每次问答一个追踪号：阶段事件与汇总行都挂着它（P2-2）
        trace_id = new_trace_id()
        initial: AgentState = {
            "question": raw_question,
            "history": history or [],
            "retry_count": 0,
            "max_retries": self.max_retries,
            "trace_id": trace_id,
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
                    trace_id=trace_id, engine="graph",
                    elapsed=round(time.perf_counter() - started, 2),
                    question=raw_question[:200], question_len=len(raw_question),
                    truncated=truncated, error=f"{type(e).__name__}: {e}",
                    llm_calls=counter.calls, tokens_in=counter.input_tokens,
                    tokens_out=counter.output_tokens)
            raise

        result = state.get("result")
        unknown = state.get("unknown_enum")
        att = state.get("attribution") or {}

        # 追问建议：纯规则计算（零 token），失败也不该影响本次回答
        try:
            if unknown:
                # 澄清场景下通用建议没意义，直接把可选值做成能点的问题
                state["suggestions"] = [
                    f"{v}的销售额是多少？"
                    for v in (unknown.get("values") or [])[:3]]
            elif att.get("ok"):
                # 归因场景：顺着"换个维度再拆"往下问，比通用建议更接得上。
                # 要排除已经用过的拆解维度和对比列本身，否则会建议"再按区域拆"
                # —— 而区域正是刚才的对比列。
                used = {att.get("dim") or "", att.get("compare_col") or ""}
                alt = next((d for d in ("区域", "品类", "客户等级") if d not in used), "区域")
                state["suggestions"] = [
                    f"同样的对比按「{alt}」拆解一下",
                    f"「{att.get('current_label')}」的{att.get('metric')}走势如何？",
                ]
            else:
                state["suggestions"] = suggest_followups(
                    state.get("rewritten_question") or raw_question, result)
        except Exception:
            state["suggestions"] = []

        log_run(
            ok=result is not None or bool(unknown) or bool(state.get("defer")),
            trace_id=trace_id,
            engine="graph",
            clarify=bool(unknown),
            defer=bool(state.get("defer")),
            attribution=bool((state.get("attribution") or {}).get("ok")),
            route=state.get("route", ""),
            retries=int(state.get("retry_count") or 0),
            rows=0 if result is None else int(len(result)),
            empty=is_empty_result(result) or is_aggregate_empty(result),
            lessons=len(state.get("lessons_used") or []),
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
