"""运行时上下文：数据、Schema、业务术语的惰性加载与共享。

为什么单独抽出来：
- LangGraph 的 state 需要可序列化/可合并，不适合塞 DataFrame 与连接配置；
- 线性编排（DataAgent）与状态机编排（graph）需要同一份数据视图，避免重复加载。

所有重资源都用 cached_property，进程内只加载一次。
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import pandas as pd

from app.schema.loader import (
    DEFAULT_DB,
    build_db_schema_text,
    build_schema_text,
    get_db_path,
    load_catalog_terms,
    load_dataframe,
)


@dataclass
class AgentContext:
    """一次会话的静态数据上下文（数据源 + 语义）。"""
    db_name: str = DEFAULT_DB
    csv_name: str = "sales_orders.csv"

    @cached_property
    def db_path(self) -> Optional[str]:
        """数据库路径；不存在则返回 None（自动降级为单表模式）。"""
        try:
            return get_db_path(self.db_name)
        except FileNotFoundError:
            return None

    @cached_property
    def db_schema(self) -> str:
        return build_db_schema_text(self.db_path) if self.db_path else ""

    @cached_property
    def terms(self) -> str:
        return load_catalog_terms()

    @cached_property
    def df(self) -> pd.DataFrame:
        return load_dataframe(self.csv_name)

    @cached_property
    def df_schema(self) -> str:
        return build_schema_text(self.df)

    @property
    def has_db(self) -> bool:
        return bool(self.db_path)

    @property
    def mode_label(self) -> str:
        return "SQLite 多表（Text-to-SQL）" if self.has_db else "单表 CSV（Text-to-Pandas）"
