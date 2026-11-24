"""运行时上下文：数据、Schema、业务术语的惰性加载与共享。

为什么单独抽出来：
- LangGraph 的 state 需要可序列化/可合并，不适合塞 DataFrame 与连接配置；
- 线性编排（DataAgent）与状态机编排（graph）需要同一份数据视图，避免重复加载。

所有重资源都用 cached_property，进程内只加载一次。
"""
from dataclasses import dataclass
from functools import cached_property
import os
from typing import Optional

import pandas as pd

from app.schema.loader import (
    DEFAULT_DB,
    build_db_schema_text,
    build_schema_text,
    collect_data_profile,
    format_data_profile,
    get_db_path,
    load_catalog,
    load_catalog_terms,
    load_dataframe,
)


@dataclass
class AgentContext:
    """一次会话的静态数据上下文（数据源 + 语义）。"""
    db_name: str = DEFAULT_DB
    csv_name: str = "sales_orders.csv"
    db_file: Optional[str] = None    # 命令行 `--db` 指定的库文件，优先于 data/ 下的默认库

    @cached_property
    def db_path(self) -> Optional[str]:
        """数据库路径；不存在则返回 None（自动降级为单表模式）。"""
        if self.db_file:
            path = os.path.abspath(os.path.expanduser(self.db_file))
            return path if os.path.exists(path) else None
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
    def data_profile(self) -> dict:
        """实测的数据画像：日期覆盖范围 + 枚举列真实取值（P0-1b）。

        与 db_schema 的分工：schema 说明"表怎么建的"，画像说明"库里实际有什么"。
        空结果复核、取值澄清都基于事实，而不是猜。
        """
        return collect_data_profile(self.db_path, self.catalog) if self.db_path else {}

    @cached_property
    def data_profile_text(self) -> str:
        return format_data_profile(self.data_profile)

    @cached_property
    def catalog(self) -> dict:
        return load_catalog()

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
