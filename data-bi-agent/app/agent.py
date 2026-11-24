"""统一编排入口（对应功能 F3.1 的轻量版，阶段 2 实现）。

职责：把"问题 → 语句生成 → 执行 → 图表 → 洞察"串成一条链，并决定走 SQL 还是 pandas。

路由策略（先 SQL 后 pandas，失败自动回退）：
- 有 SQLite 数据库时优先 Text-to-SQL：多表 JOIN、聚合下推都由数据库完成，结果最准；
- SQL 生成或执行失败时，回退 Text-to-Pandas（单表宽表），保证"总能给个答案"；
- 也可用 force_pandas=True 强制走 pandas。

阶段 3 会把这条链替换为 LangGraph 状态机（含反思重试），当前实现保持线性、可读。
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

from app.context import AgentContext
from app.nodes.answer import generate_insight, local_insight, summarize_result
from app.nodes.execute import execute_code
from app.nodes.generate import generate_code, generate_sql
from app.schema.loader import DEFAULT_DB
from app.tools.chart_tool import CHART_TYPE_LABEL, build_chart, save_chart
from app.tools.sql_tool import run_sql


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
    """

    def __init__(self, db_name: str = DEFAULT_DB, csv_name: str = "sales_orders.csv",
                 ctx: Optional[AgentContext] = None):
        # ctx 用于命令行 `--db` 这类自定义数据源；不给则按 db_name 走默认目录
        self.ctx = ctx or AgentContext(db_name=db_name, csv_name=csv_name)

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
    def ask(self, question: str, with_chart: bool = True,
            with_insight: bool = True, force_pandas: bool = False) -> AskResult:
        res = AskResult(question=question)

        if self.db_path and not force_pandas:
            self._try_sql(question, res)

        if res.df is None:
            self._try_pandas(question, res)

        if res.df is not None:
            res.summary = summarize_result(res.df)
            if with_chart:
                self._attach_chart(question, res)
            if with_insight:
                res.insight = generate_insight(question, res.df)

        return res

    def _try_sql(self, question: str, res: AskResult) -> None:
        try:
            sql = generate_sql(self.db_schema, self.terms, question)
        except Exception as e:
            res.errors.append(f"SQL 生成失败：{type(e).__name__}: {e}")
            return

        r = run_sql(sql, self.db_path)
        if r["ok"]:
            res.route, res.statement, res.df = "sql", r["sql"], r["df"]
        else:
            res.errors.append(f"SQL {r['stage']} 阶段失败：{r['error']}")

    def _try_pandas(self, question: str, res: AskResult) -> None:
        try:
            code = generate_code(self.df_schema, self.terms, question)
        except Exception as e:
            res.errors.append(f"代码生成失败：{type(e).__name__}: {e}")
            return

        r = execute_code(code, self.df)
        if r["ok"]:
            res.route, res.statement = "pandas", code
            res.df = to_frame(r["result"])
        else:
            res.errors.append(f"代码执行失败（{r['stage']}）：{r['error']}")

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
