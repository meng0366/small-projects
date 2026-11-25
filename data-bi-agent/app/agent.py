"""统一编排入口（对应功能 F3.1 的轻量版，阶段 2 实现）。

职责：把"问题 → 语句生成 → 执行 → 图表 → 洞察"串成一条链，并决定走 SQL 还是 pandas。

路由策略（先 SQL 后 pandas，失败自动回退）：
- 有 SQLite 数据库时优先 Text-to-SQL：多表 JOIN、聚合下推都由数据库完成，结果最准；
- SQL 生成或执行失败时，回退 Text-to-Pandas（单表宽表），保证"总能给个答案"；
- 也可用 force_pandas=True 强制走 pandas。

阶段 3 会把这条链替换为 LangGraph 状态机（含反思重试），当前实现保持线性、可读。

链路追踪（P2-2）：线性版也要能被 `logs --trace` 看到，否则两种引擎排查方式不一致。
这里按状态机同名阶段（generate / execute / answer）写阶段事件，
SQL 失败回退 pandas 时会再出现一组 generate / execute，链里一眼能看出"回退过"。
"""
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

from app.context import AgentContext
from app.logging_utils import log_event, log_run, new_trace_id, trace_fields
from app.nodes.answer import generate_insight, local_insight, summarize_result
from app.nodes.execute import execute_code
from app.nodes.generate import generate_code, generate_sql
from app.nodes.plan import build_plan
from app.nodes.reflect import is_aggregate_empty, is_empty_result
from app.schema.loader import DEFAULT_DB
from app.tools.chart_tool import CHART_TYPE_LABEL, build_chart, save_chart
from app.tools.enum_guard import build_clarify_answer
from app.tools.peek import MAX_PEEKS, PEEK_OFF_NOTE, maybe_peek
from app.tools.sql_cache import (
    find_cached_sql,
    forget_sql,
    remember_sql,
    result_fingerprint,
)
from app.tools.time_anchor import format_time_hint, out_of_range_answer
from app.tools.sql_tool import run_sql
from app.tools.suggest_tool import suggest_followups


@dataclass
class AskResult:
    """一次问答的完整产出。"""
    question: str
    route: str = ""                 # "sql" | "pandas"
    statement: str = ""             # 实际执行的 SQL 或 Python 代码
    df: Optional[pd.DataFrame] = None
    insight: str = ""
    chart_path: str = ""
    chart_type: str = "none"
    summary: str = ""
    trace_id: str = ""              # 本次问答的追踪号（P2-2），阶段事件与汇总行共用
    plan: Optional[dict] = None     # 结构化查询计划（P1），None 表示这次没出计划
    plan_issues: List[dict] = field(default_factory=list)
    peek: dict = field(default_factory=dict)   # 两段式探查（P1）：used/rows/note/error/ms
    errors: List[str] = field(default_factory=list)
    refuse_reason: str = ""         # 理解阶段直接拒答的原因（P0 时间锚点），无生成无执行
    # SQL 复用（P1）：命中时 statement 来自缓存，这次没有"生成"这一步
    cache_hit: bool = False
    cache_note: str = ""
    cached_insight: str = ""        # 上次的洞察文案（结果指纹一致时才填）

    @property
    def ok(self) -> bool:
        return self.df is not None

    @property
    def chart_label(self) -> str:
        return CHART_TYPE_LABEL.get(self.chart_type, self.chart_type)


def to_frame(result: Any) -> pd.DataFrame:
    """把 Series / 标量统一转成 DataFrame，便于画图与摘要。"""
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.reset_index()
    return pd.DataFrame({"结果": [result]})


class DataAgent:
    """数据分析 Agent 的门面类（线性编排版）。

    状态机编排见 app/graph.py，两者共用同一个 AgentContext。
    对外接口刻意与 GraphAgent 对齐（`run(question, history, config)`），
    这样 CLI 的 `ask` / `chat` 不必为两种引擎写两套分支。
    """

    def __init__(self, db_name: str = DEFAULT_DB, csv_name: str = "sales_orders.csv",
                 ctx: Optional[AgentContext] = None, with_chart: bool = True,
                 with_insight: bool = True, force_pandas: bool = False):
        # ctx 用于命令行 `--db` 这类自定义数据源；不给则按 db_name 走默认目录
        self.ctx = ctx or AgentContext(db_name=db_name, csv_name=csv_name)
        self.with_chart = with_chart
        self.with_insight = with_insight
        self.force_pandas = force_pandas

    # ---- 上下文透传（保持既有调用方式不变）----
    @property
    def db_path(self):
        return self.ctx.db_path

    @property
    def db_schema(self) -> str:
        return self.ctx.db_schema

    @property
    def terms(self) -> str:
        return self.ctx.terms

    @property
    def df(self) -> pd.DataFrame:
        return self.ctx.df

    @property
    def df_schema(self) -> str:
        return self.ctx.df_schema

    @property
    def mode_label(self) -> str:
        return self.ctx.mode_label

    # ---- 主流程 ----
    def run(self, question: str, history: Optional[list] = None,
            config: Optional[dict] = None) -> dict:
        """执行一次问答，返回与 GraphAgent.run 同形状的 state dict。

        为什么要有这个方法：`chat` 只认 `agent.run()`，线性引擎原先只有 `ask()`，
        于是 `python -m app.cli chat --linear` 会直接 AttributeError。
        线性版没有多轮记忆与重试，`history` / `config` 收下但不使用（保持签名一致）。
        """
        res = self.ask(question, with_chart=self.with_chart,
                       with_insight=self.with_insight,
                       force_pandas=self.force_pandas)
        state = {
            "question": question,
            "route": res.route,
            "statement": res.statement,
            "result": res.df,
            "insight": res.insight,
            "chart_path": res.chart_path,
            "chart_type": res.chart_type,
            "errors": list(res.errors),
            "retry_count": 0,
            "trace_id": res.trace_id,
            "plan": res.plan,
            "plan_issues": list(res.plan_issues),
            "plan_used": bool(res.plan),
            "peek_used": bool(res.peek.get("peek_used")),
            "peek_note": res.peek.get("peek_note", ""),
            "peek_rows": int(res.peek.get("peek_rows") or 0),
            "peek_error": res.peek.get("peek_error", ""),
            "peek_ms": int(res.peek.get("peek_ms") or 0),
            "cache_hit": bool(res.cache_hit),
            "cache_note": res.cache_note,
            "answer": (res.refuse_reason or ("" if res.ok else "未能完成分析。")),
            "insight": res.refuse_reason or res.insight,
            "suggestions": [],
        }
        try:
            state["suggestions"] = suggest_followups(question, res.df)
        except Exception:
            pass          # 建议是锦上添花，算不出来不该影响这次回答
        return state

    def ask(self, question: str, with_chart: bool = True,
            with_insight: bool = True, force_pandas: bool = False,
            trace_id: str = "") -> AskResult:
        started = time.perf_counter()
        trace = trace_id or new_trace_id()
        res = AskResult(question=question, trace_id=trace)

        # 相对时间锚点（P0）：与状态机同一套判定。今天由系统给，
        # 完全没交集就直接说明，绝不让模型"换个有数据的月份"来答。
        anchors = self.ctx.time_anchors(question)
        refuse = out_of_range_answer(anchors, self.ctx.date_coverage)
        if refuse:
            res.refuse_reason = refuse
            log_event(trace, "answer", ms=0, failed=True)
            log_run(ok=False, trace_id=trace, engine="linear", route="refuse",
                    retries=0, rows=0, empty=False, chart="none",
                    elapsed=round(time.perf_counter() - started, 2),
                    question=(question or "")[:200], question_len=len(question or ""),
                    errors=0, error="时间超出数据覆盖范围")
            return res

        # 有交集时把锚点拼进术语，随 schema 一起进提示词（两条引擎一致）。
        hint = format_time_hint(anchors, today=self.ctx.today,
                                coverage=self.ctx.date_coverage)
        terms = (self.terms + "\n\n" + hint).strip() if hint else self.terms

        use_sql = bool(self.db_path) and not force_pandas

        # SQL 复用（P1）：命中就直接执行历史 SQL，跳过计划与生成。
        # 刻意放在计划之前 —— 计划要花一次调用，命中时先花这个钱就等于白省。
        # 跑不通（表/字段变了）会把条目作废并返回 False，照常走下面的流程。
        reused = False
        if use_sql and self.ctx.use_cache:
            entry, note = find_cached_sql(question, self.ctx)
            if entry:
                reused = self._run_cached(question, res, entry, note)

        plan_update = {}
        plan_hint = ""
        if not reused:
            # 计划先行（P1）：与状态机走同一个 build_plan，不另写一套判断。
            # 线性版没有 clarify 节点，所以三类致命问题都直接转成"说明为什么不给数"。
            plan_started = time.perf_counter()
            plan_update = build_plan(
                self.ctx, question, route="sql" if use_sql else "pandas",
                schema_info=self.db_schema if use_sql else self.df_schema,
                terms=terms, anchors=anchors)
            res.plan = plan_update.get("plan")
            res.plan_issues = list(plan_update.get("plan_issues") or [])

            blocked = (plan_update.get("time_defer") or plan_update.get("defer")
                       or (build_clarify_answer(plan_update["unknown_enum"])
                           if plan_update.get("unknown_enum") else ""))
            if blocked:
                res.refuse_reason = blocked
                log_event(trace, "plan",
                          ms=round((time.perf_counter() - plan_started) * 1000),
                          **trace_fields({"plan_reason": plan_update.get("plan_reason"),
                                          "plan_issues": res.plan_issues,
                                          "time_defer": plan_update.get("time_defer")},
                                         failed=True))
                log_event(trace, "answer", ms=0, failed=True)
                log_run(ok=False, trace_id=trace, engine="linear", route="refuse",
                        retries=0, rows=0, empty=False, chart="none",
                        plan=plan_update.get("plan_reason", ""),
                        plan_issues=len(res.plan_issues),
                        elapsed=round(time.perf_counter() - started, 2),
                        question=(question or "")[:200], question_len=len(question or ""),
                        errors=0, error="计划校验未通过")
                return res
            # 只在**真的出了计划**时记阶段事件：跳过的计划不是阶段，
            # 它写在汇总行的 plan 字段里（与状态机一致 —— 那边没走到 plan 节点就没这行）
            if res.plan:
                log_event(trace, "plan",
                          ms=round((time.perf_counter() - plan_started) * 1000),
                          **trace_fields({"plan_reason": plan_update.get("plan_reason"),
                                          "plan_issues": res.plan_issues}))
            plan_hint = plan_update.get("plan_hint") or ""

        if use_sql and res.df is None:
            self._try_sql(question, res, terms=terms, plan_hint=plan_hint)

        # SQL 失败才回退 pandas；回退这件事本身要写进链里，否则看不出"为什么慢/为什么换了路线"
        fallback = res.df is None and bool(self.db_path) and not force_pandas
        if res.df is None:
            self._try_pandas(question, res, fallback=fallback, terms=terms,
                             plan_hint=plan_hint)

        if res.df is not None:
            answered = time.perf_counter()
            res.summary = summarize_result(res.df)
            if with_chart:
                self._attach_chart(question, res)
            if with_insight:
                # 结果指纹与上次一致时直接沿用结论 —— 这是整次问答里
                # 复用时唯一还会剩下的一次模型调用，省掉它才是真的零成本。
                res.insight = res.cached_insight or generate_insight(question, res.df)
            self._remember(question, res)
            log_event(trace, "answer",
                      ms=round((time.perf_counter() - answered) * 1000),
                      **trace_fields({"result": res.df, "chart_type": res.chart_type}))
        else:
            # 两条路都没成：链尾留一行失败的 answer，否则链看起来"没走完"
            log_event(trace, "answer", ms=0,
                      **trace_fields({"error": res.errors[-1] if res.errors else "没有拿到结果"},
                                     failed=True))

        log_run(
            ok=res.ok,
            trace_id=trace,
            engine="linear",           # 与状态机汇总行区分：线性版没有 token 回调
            route=res.route,
            retries=0,
            rows=0 if res.df is None else int(len(res.df)),
            empty=is_empty_result(res.df) or is_aggregate_empty(res.df),
            chart=res.chart_type,
            plan=plan_update.get("plan_reason", ""),
            plan_issues=len(res.plan_issues),
            peek_used=bool(res.peek.get("peek_used")),
            peek_rows=int(res.peek.get("peek_rows") or 0),
            peek_error=(res.peek.get("peek_error") or "")[:80],
            cache=bool(res.cache_hit),
            cache_note=(res.cache_note or "")[:40],
            elapsed=round(time.perf_counter() - started, 2),
            question=(question or "")[:200],
            question_len=len(question or ""),
            errors=len(res.errors),
            error=(res.errors[-1][:300] if res.errors else ""),
        )
        return res

    def _run_cached(self, question: str, res: AskResult, entry: dict, note: str) -> bool:
        """执行复用的历史 SQL；跑不通就作废这条缓存并返回 False（调用方重新生成）。

        为什么失败要作废：缓存里的 SQL 对不上当前库（表被删、字段改名），
        留着它只会每次命中、每次失败、每次再重新生成，白搭一次执行。
        """
        trace = res.trace_id
        sql = str(entry.get("sql") or "")
        started = time.perf_counter()
        r = run_sql(sql, self.db_path)
        if not r["ok"]:
            forget_sql(question, self.ctx)
            msg = f"复用的 SQL 已失效（{r['stage']} 阶段）：{r['error']}"
            res.errors.append(msg)
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "sql", "error": msg},
                                     cache_failed=True))
            return False

        res.route, res.statement, res.df = "sql", r["sql"], r["df"]
        res.cache_hit = True
        res.cache_note = note

        # 数据还是不是上次那份：只有指纹一致才敢沿用上次的结论
        cached_hash = entry.get("result_fp") or ""
        if cached_hash and result_fingerprint(res.df) == cached_hash:
            res.cached_insight = str(entry.get("insight") or "")
        else:
            res.cache_note = "查询已复用，但数据有变化，结论重新生成"

        log_event(trace, "execute",
                  ms=round((time.perf_counter() - started) * 1000),
                  **trace_fields({"route": "sql", "result": res.df,
                                  "cache_note": res.cache_note}, cache_hit=True))
        return True

    def _remember(self, question: str, res: AskResult) -> None:
        """把跑通的「问题 → SQL」存进缓存（只存真的给出答案的这次）。"""
        if not self.ctx.use_cache or res.route != "sql" or not res.statement:
            return
        try:
            remember_sql(question, res.statement, self.ctx,
                         df=res.df, insight=res.insight)
        except Exception:
            pass          # 缓存是加速手段，写不进去不能影响这次回答

    def _try_sql(self, question: str, res: AskResult, terms: str = None,
                 plan_hint: str = "") -> None:
        trace = res.trace_id
        hint = plan_hint
        if not self.ctx.use_peek:
            hint = (hint + "\n\n" + PEEK_OFF_NOTE).strip()

        def _generate(extra: str = "") -> str:
            merged = (hint + "\n\n" + extra).strip() if hint else extra
            return generate_sql(self.db_schema, terms or self.terms, question,
                                plan_hint=merged)

        started = time.perf_counter()
        try:
            sql = _generate()
        except Exception as e:
            msg = f"SQL 生成失败：{type(e).__name__}: {e}"
            res.errors.append(msg)
            log_event(trace, "generate",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "sql", "error": msg}, failed=True))
            return

        # 两段式探查（P1）：与状态机共用 maybe_peek，流程只有一份实现
        sql, peek_update = maybe_peek(self.ctx, sql, lambda note: _generate(note),
                                      trace_id=trace, remaining=MAX_PEEKS)
        res.peek = peek_update
        if not sql:
            msg = peek_update.get("peek_error") or "生成失败：没有拿到可执行语句"
            res.errors.append(msg)
            log_event(trace, "generate",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "sql", "error": msg, **peek_update},
                                     failed=True))
            return

        # 语句原文不落盘，只记长度 —— 白名单在 logging_utils 里统一管这件事
        log_event(trace, "generate",
                  ms=round((time.perf_counter() - started) * 1000),
                  **trace_fields({"route": "sql", "statement": sql, **peek_update}))

        started = time.perf_counter()
        r = run_sql(sql, self.db_path)
        if r["ok"]:
            res.route, res.statement, res.df = "sql", r["sql"], r["df"]
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "sql", "result": r["df"]}))
        else:
            msg = f"SQL {r['stage']} 阶段失败：{r['error']}"
            res.errors.append(msg)
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "sql", "error": msg}, failed=True))

    def _try_pandas(self, question: str, res: AskResult, fallback: bool = False,
                    terms: str = None, plan_hint: str = "") -> None:
        trace = res.trace_id
        started = time.perf_counter()
        try:
            code = generate_code(self.df_schema, terms or self.terms, question,
                                 plan_hint=plan_hint)
        except Exception as e:
            msg = f"代码生成失败：{type(e).__name__}: {e}"
            res.errors.append(msg)
            log_event(trace, "generate",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "pandas", "error": msg},
                                     failed=True, fallback=fallback))
            return
        log_event(trace, "generate",
                  ms=round((time.perf_counter() - started) * 1000),
                  **trace_fields({"route": "pandas", "statement": code},
                                 fallback=fallback))

        started = time.perf_counter()
        r = execute_code(code, self.df)
        if r["ok"]:
            res.route, res.statement = "pandas", code
            res.df = to_frame(r["result"])
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "pandas", "result": res.df},
                                     fallback=fallback))
        else:
            msg = f"代码执行失败（{r['stage']}）：{r['error']}"
            res.errors.append(msg)
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      **trace_fields({"route": "pandas", "error": msg},
                                     failed=True, fallback=fallback))

    def _attach_chart(self, question: str, res: AskResult) -> None:
        try:
            fig, chart_type = build_chart(res.df, question)
            res.chart_type = chart_type
            if fig is not None:
                res.chart_path = save_chart(fig)
        except Exception as e:
            res.errors.append(f"图表生成失败：{type(e).__name__}: {e}")

    # ---- 便捷方法 ----
    def explain(self, question: str) -> str:
        """只要洞察文案（供 UI 或脚本调用）。"""
        res = self.ask(question, with_chart=False, with_insight=True)
        return res.insight or local_insight(res.df)
