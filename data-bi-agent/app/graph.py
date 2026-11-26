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
from app.logging_utils import TokenCounter, log_event, log_run, new_trace_id, trace_fields
from app.nodes.answer import generate_insight, local_insight
from app.nodes.cache import make_cache_node
from app.nodes.execute import execute_code
from app.nodes.generate import generate_attribution_sql, generate_code, generate_sql
from app.nodes.plan import make_plan_node
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
from app.tools.peek import MAX_PEEKS, PEEK_OFF_NOTE, maybe_peek
from app.tools.scope_guard import check_scope, looks_like_sql, parse_defer
from app.tools.sql_cache import forget_sql, remember_sql, result_fingerprint
from app.tools.sql_review import format_review_for_model, review_sql
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
        # 计划先行（P1）：计划已通过代码校验，拼在问题后面一起下发。
        # 重试时**沿用同一份计划** —— 空结果重试不许改口径，计划正是那条口径。
        plan_hint = state.get("plan_hint") or ""
        if not ctx.use_peek:
            # 关掉探查时靠"紧贴问题的指令"拦住模型，而不是事后丢弃它的 PEEK 行：
            # 后者会让模型白要一次、我们再干耗一次调用重新生成。
            plan_hint = (plan_hint + "\n\n" + PEEK_OFF_NOTE).strip()

        def _generate(hint: str = plan_hint) -> str:
            if intent and route == "sql":
                # 归因问题要的不是"一个总数"，而是能拆的交叉表（P1-2）
                return generate_attribution_sql(
                    state.get("schema_info", ""), state.get("terms", ""), question, intent,
                    previous_error=previous_error, previous_empty=previous_empty,
                    lessons=lessons, plan_hint=hint)
            if route == "sql":
                return generate_sql(state.get("schema_info", ""), state.get("terms", ""),
                                    question, previous_error=previous_error,
                                    previous_empty=previous_empty, lessons=lessons,
                                    plan_hint=hint)
            return generate_code(state.get("schema_info", ""), state.get("terms", ""),
                                 question, previous_error=previous_error,
                                 previous_empty=previous_empty, lessons=lessons,
                                 plan_hint=hint)

        try:
            statement = _generate()
        except Exception as e:
            return {"statement": "", "error": f"生成失败：{type(e).__name__}: {e}",
                    "lessons_used": used}

        # 两段式探查（P1）：模型若先要"看一眼真实取值"，这里执行一次并把结果回灌，
        # 再让它写最终 SQL。额度按整次问答算（`peek_used` 进 state），重试时不再探。
        statement, peek_update = maybe_peek(
            ctx, statement,
            lambda note: _generate((plan_hint + "\n\n" + note).strip() if plan_hint else note),
            trace_id=state.get("trace_id", ""),
            remaining=0 if state.get("peek_used") else MAX_PEEKS)

        # 审查结论属于"上一条语句"：换成新语句时先清掉，免得重试成功后还挂着旧提示
        update = {**peek_update, "statement": statement, "lessons_used": used,
                  "review_issues": [], "review_blocked": False}
        if not statement:
            # 探查后重新生成失败：交给重试环，绝不能把 `PEEK:` 那行当 SQL 去执行
            update["error"] = (peek_update.get("peek_error")
                               or "生成失败：没有拿到可执行语句")
        return update

    return node


def _reuse_update(state: AgentState, frame) -> dict:
    """复用时判断"这次跑出来的还是不是上次那份数据"，不是就丢掉旧结论。

    SQL 复用是安全的：同一个问题、同一个库结构、同一个参考日，查询本身不变。
    但**洞察文案依赖数据**，而库是会变的（新增订单、补录数据）。所以缓存里
    同时存了结果指纹 —— 只有数据一字不差才敢沿用上次的结论，否则保留 SQL、
    重新让模型看一眼新数据。

    指纹算不出来（结果里有不可哈希的嵌套值）时按"不一样"处理：多花一次调用，
    好过把一份过期的结论当新的给出去。
    """
    if not state.get("cache_hit"):
        return {}
    cached = state.get("cached_hash") or ""
    current = result_fingerprint(frame)
    if cached and current and cached == current:
        return {}
    return {"cached_insight": "", "cached_hash": current,
            "cache_note": "查询已复用，但数据有变化，结论重新生成"}


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
            # 结构审查放在所有入口的汇合处：模型现写的、缓存里复用的、人工 teach 的
            # 都要过这一关。纯函数、零 token；block 不执行、把原因回喂给模型改写，
            # warn 只写进 state 供 CLI / 网页展示（可能是对的，拦错比漏判更伤）。
            review_update = {}
            if ctx.use_review:
                try:
                    review = review_sql(statement, ctx.db_unique_keys)
                    if review["issues"]:
                        review_update["review_issues"] = review["issues"]
                    if not review["ok"]:
                        return {"result": None,
                                "error": format_review_for_model(review["issues"]),
                                "review_blocked": True, **review_update, **history}
                except Exception:
                    pass  # 审查器是加法，自身异常不能挡住本来能跑的查询
            r = run_sql(statement, ctx.db_path)
            if r["ok"]:
                frame = r["df"]
                problems = check_scope(question, statement, ctx.enums) \
                    if retrying_after_empty and not is_empty_result(frame) else []
                if problems:
                    # 拿到了数据，但是靠改动用户的条件换来的（如 2024→2025）→ 丢掉，不许用
                    return {"result": None, "defer": "；".join(problems),
                            "scope_violation": problems, **review_update, **history}
                update = {"result": frame, "error": None,
                          "empty_result": is_empty_result(frame),
                          **review_update, **history}
                update.update(_reuse_update(state, frame))
                return update
            if state.get("cache_hit"):
                # 复用的 SQL 跑不通（表被删、字段改名）→ 立刻作废这条缓存。
                # 同时标记 cache_failed：这次失败是"缓存过时"，不是"模型理解错"，
                # reflect 不该因此换到 pandas 路线（那会绕过全部 SQL 能力）。
                forget_sql(state.get("question", ""), ctx)
                return {"result": None, "cache_hit": False, "cache_failed": True,
                        "cached_insight": "", "cached_hash": "",
                        "cache_note": "历史 SQL 已失效，重新生成",
                        "error": f"SQL {r['stage']} 阶段：{r['error']}",
                        "empty_result": False, **history}
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


def _remember(ctx: AgentContext, state: AgentState, result, insight: str) -> None:
    """把"跑通并给出答案"的这次查询存进缓存（P1 SQL 复用）。

    只在这里写，不在 execute 里写：execute 成功只说明"这条 SQL 能跑"，
    不代表这轮问答真的给了答案（可能之后被判空结果、被判拒答）。
    空结果与拒答固化下来最危险 —— 数据一更新，用户永远只能看到"没数据"。

    多轮追问（history 非空）一律不写：缓存键是问题原文，而同一句话在不同
    上下文里指向的可能是不同的东西，写进去会被下一次单轮提问误命中。
    """
    if not getattr(ctx, "use_cache", True):
        return
    if state.get("route") != "sql" or state.get("attribution_intent"):
        return
    if state.get("history"):
        return
    try:
        remember_sql(state.get("question", ""), state.get("statement", ""), ctx,
                     df=result, insight=insight)
    except Exception:
        pass          # 缓存是加速手段，写不进去不能影响这次回答


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

        # 洞察文案复用（P1）：结果指纹一致才用（`_reuse_update` 已经判过），
        # 这一条省掉的是整次问答里最后也是唯一剩下的模型调用。
        cached = (state.get("cached_insight") or "").strip()
        if with_insight and cached:
            insight = cached
            update["cache_note"] = "查询与结果都与上次一致，结论一并复用"
        elif with_insight:
            insight = generate_insight(question, result)
        else:
            insight = local_insight(result)

        _remember(ctx, state, result, insight)

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
        # 相对时间完全落在数据覆盖之外（P0）：直接说明没有数据。
        # 与取值澄清共用这个终止节点，都不该进生成 —— 让模型去"放宽时间"
        # 正是 P0-2b 拿回别的年份数据冒充的起点。
        time_defer = (state.get("time_defer") or "").strip()
        # 计划校验发现"计划里的年份/月份与用户原话不一致"时，也走这个终止节点：
        # 不给数的理由必须说清楚，而不是让生成阶段去"改一改就查得出来"（P1）
        plan_defer = (state.get("defer") or "").strip()
        if time_defer:
            answer = time_defer
        elif plan_defer:
            answer = plan_defer
        else:
            answer = build_clarify_answer(state.get("unknown_enum") or {})
        return {"answer": answer, "insight": answer,
                "chart_type": "none", "chart_path": "", "result": None}

    return node


def route_after_cache(state: AgentState) -> str:
    """条件边：命中缓存就直接执行，否则照常理解问题（P1 SQL 复用）。"""
    return "execute" if state.get("cache_hit") and state.get("statement") else "understand"


def route_after_understand(state: AgentState) -> str:
    """条件边：取值/时间对不上先澄清；需要先想清楚就先出计划，否则直接生成。"""
    if state.get("time_defer") or state.get("unknown_enum"):
        return "clarify"
    return "plan" if state.get("plan_needed") else "generate"


def route_after_plan(state: AgentState) -> str:
    """条件边：计划校验出致命问题就短路澄清，否则带着计划去生成（P1）。

    致命问题只有三类（时间无交集、取值不存在、年份被换），都由代码判定；
    表名列名对不上只算提醒，照常生成 —— 误拦比漏拦更伤。
    """
    if state.get("time_defer") or state.get("unknown_enum") or state.get("defer"):
        return "clarify"
    return "generate"


# ---------------------------------------------------------------- 链路追踪（P2-2）
#
# 汇总行只说"最后怎么样"，看不出"慢在哪一步、重试发生在哪、错在哪一层"。
# 这里给每个节点包一层：执行完补一行阶段事件，全部共用同一个 trace_id。
# 记哪些字段由 `logging_utils.trace_fields()` 统一决定（白名单在那儿，
# 线性编排也用它）—— 绝不能把结果表或 SQL 全文写进日志。


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
                  **trace_fields(update))
        return update

    return wrapped


def build_graph(ctx: Optional[AgentContext] = None, with_chart: bool = True,
                with_insight: bool = True, max_retries: int = DEFAULT_MAX_RETRIES):
    """编译状态机，返回可直接 invoke 的 graph。"""
    ctx = ctx or AgentContext()

    builder = StateGraph(AgentState)
    builder.add_node("cache", _traced("cache", make_cache_node(ctx)))
    builder.add_node("understand", _traced("understand", make_understand_node(ctx)))
    builder.add_node("plan", _traced("plan", make_plan_node(ctx)))
    builder.add_node("clarify", _traced("clarify", make_clarify_node(ctx)))
    builder.add_node("generate", _traced("generate", make_generate_node(ctx)))
    builder.add_node("execute", _traced("execute", make_execute_node(ctx)))
    builder.add_node("reflect", _traced("reflect", make_reflect_node(ctx)))
    builder.add_node("answer", _traced("answer",
                                      make_answer_node(ctx, with_chart, with_insight)))

    builder.add_edge(START, "cache")
    builder.add_conditional_edges(
        "cache", route_after_cache,
        {"execute": "execute", "understand": "understand"})
    builder.add_conditional_edges(
        "understand", route_after_understand,
        {"plan": "plan", "generate": "generate", "clarify": "clarify"})
    builder.add_conditional_edges(
        "plan", route_after_plan,
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
            plan=state.get("plan_reason", ""),
            plan_issues=len(state.get("plan_issues") or []),
            peek_used=bool(state.get("peek_used")),
            peek_rows=int(state.get("peek_rows") or 0),
            peek_error=(state.get("peek_error") or "")[:80],
            cache=bool(state.get("cache_hit")),
            cache_note=(state.get("cache_note") or "")[:40],
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
