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
from app.logging_utils import log_event, log_run, new_trace_id
from app.nodes.answer import generate_insight, local_insight, summarize_result
from app.nodes.execute import execute_code
from app.nodes.generate import generate_code, generate_sql
from app.nodes.reflect import is_aggregate_empty, is_empty_result
from app.schema.loader import DEFAULT_DB
from app.tools.chart_tool import CHART_TYPE_LABEL, build_chart, save_chart
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
    errors: List[str] = field(default_factory=list)

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
            "answer": "" if res.ok else "未能完成分析。",
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

        if self.db_path and not force_pandas:
            self._try_sql(question, res)

        # SQL 失败才回退 pandas；回退这件事本身要写进链里，否则看不出"为什么慢/为什么换了路线"
        fallback = res.df is None and bool(self.db_path) and not force_pandas
        if res.df is None:
            self._try_pandas(question, res, fallback=fallback)

        if res.df is not None:
            answered = time.perf_counter()
            res.summary = summarize_result(res.df)
            if with_chart:
                self._attach_chart(question, res)
            if with_insight:
                res.insight = generate_insight(question, res.df)
            log_event(trace, "answer",
                      ms=round((time.perf_counter() - answered) * 1000),
                      rows=len(res.df), chart=res.chart_type or "")
        else:
            # 两条路都没成：链尾留一行失败的 answer，否则链看起来"没走完"
            log_event(trace, "answer", ms=0, failed=True,
                      error=(res.errors[-1][:200] if res.errors else "没有拿到结果"))

        log_run(
            ok=res.ok,
            trace_id=trace,
            engine="linear",           # 与状态机汇总行区分：线性版没有 token 回调
            route=res.route,
            retries=0,
            rows=0 if res.df is None else int(len(res.df)),
            empty=is_empty_result(res.df) or is_aggregate_empty(res.df),
            chart=res.chart_type,
            elapsed=round(time.perf_counter() - started, 2),
            question=(question or "")[:200],
            question_len=len(question or ""),
            errors=len(res.errors),
            error=(res.errors[-1][:300] if res.errors else ""),
        )
        return res

    def _try_sql(self, question: str, res: AskResult) -> None:
        trace = res.trace_id
        started = time.perf_counter()
        try:
            sql = generate_sql(self.db_schema, self.terms, question)
        except Exception as e:
            msg = f"SQL 生成失败：{type(e).__name__}: {e}"
            res.errors.append(msg)
            log_event(trace, "generate",
                      ms=round((time.perf_counter() - started) * 1000),
                      route="sql", failed=True, error=msg[:200])
            return
        # 语句原文不落盘，只记长度 —— 日志会被翻、会被分享
        log_event(trace, "generate",
                  ms=round((time.perf_counter() - started) * 1000),
                  route="sql", statement_len=len(sql))

        started = time.perf_counter()
        r = run_sql(sql, self.db_path)
        if r["ok"]:
            res.route, res.statement, res.df = "sql", r["sql"], r["df"]
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      route="sql", rows=len(r["df"]))
        else:
            msg = f"SQL {r['stage']} 阶段失败：{r['error']}"
            res.errors.append(msg)
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      route="sql", failed=True, error=msg[:200])

    def _try_pandas(self, question: str, res: AskResult, fallback: bool = False) -> None:
        trace = res.trace_id
        started = time.perf_counter()
        try:
            code = generate_code(self.df_schema, self.terms, question)
        except Exception as e:
            msg = f"代码生成失败：{type(e).__name__}: {e}"
            res.errors.append(msg)
            log_event(trace, "generate",
                      ms=round((time.perf_counter() - started) * 1000),
                      route="pandas", failed=True, error=msg[:200],
                      **({"fallback": True} if fallback else {}))
            return
        log_event(trace, "generate",
                  ms=round((time.perf_counter() - started) * 1000),
                  route="pandas", statement_len=len(code),
                  **({"fallback": True} if fallback else {}))

        started = time.perf_counter()
        r = execute_code(code, self.df)
        if r["ok"]:
            res.route, res.statement = "pandas", code
            res.df = to_frame(r["result"])
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      route="pandas", rows=len(res.df),
                      **({"fallback": True} if fallback else {}))
        else:
            msg = f"代码执行失败（{r['stage']}）：{r['error']}"
            res.errors.append(msg)
            log_event(trace, "execute",
                      ms=round((time.perf_counter() - started) * 1000),
                      route="pandas", failed=True, error=msg[:200],
                      **({"fallback": True} if fallback else {}))

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
