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

from app.nodes.answer import generate_insight, local_insight, summarize_result
from app.nodes.execute import execute_code
from app.nodes.generate import generate_code, generate_sql
from app.schema.loader import (
    DEFAULT_DB,
    build_db_schema_text,
    build_schema_text,
    get_db_path,
    load_catalog_terms,
    load_dataframe,
)
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
    """数据分析 Agent 的门面类。"""

    def __init__(self, db_name: str = DEFAULT_DB, csv_name: str = "sales_orders.csv"):
        self.csv_name = csv_name
        self.terms = load_catalog_terms()
        self._df: Optional[pd.DataFrame] = None
        self._df_schema: Optional[str] = None

        # 数据库可用性：缺失时自动退回 pandas 单表模式，不影响启动
        self.db_path: Optional[str] = None
        self.db_schema: str = ""
        try:
            self.db_path = get_db_path(db_name)
            self.db_schema = build_db_schema_text(self.db_path)
        except FileNotFoundError:
            pass

    # ---- 惰性加载宽表 ----
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = load_dataframe(self.csv_name)
        return self._df

    @property
    def df_schema(self) -> str:
        if self._df_schema is None:
            self._df_schema = build_schema_text(self.df)
        return self._df_schema

    @property
    def mode_label(self) -> str:
        return "SQLite 多表（Text-to-SQL）" if self.db_path else "单表 CSV（Text-to-Pandas）"

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
