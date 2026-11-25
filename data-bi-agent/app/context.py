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
    describe_tables,
    format_data_profile,
    get_db_path,
    load_catalog,
    load_catalog_terms,
    load_dataframe,
)
from app.tools.lesson_store import known_words as lesson_known_words
from app.tools.lesson_store import load_lessons
from app.tools.time_anchor import resolve_time_anchor


@dataclass
class AgentContext:
    """一次会话的静态数据上下文（数据源 + 语义）。"""
    db_name: str = DEFAULT_DB
    csv_name: str = "sales_orders.csv"
    db_file: Optional[str] = None    # 命令行 `--db` 指定的库文件，优先于 data/ 下的默认库
    use_lessons: bool = True         # 是否加载修正回流（few-shot）；评测基线用 False 保证可比
    use_plan: bool = True            # 是否先出查询计划（P1）；计划会多一次模型调用，可关掉省钱

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
    def db_columns(self) -> dict:
        """`{表名: [列名]}`，供计划校验判断"计划里提到的表/列是否真的存在"。

        直接读库结构而不是解析 schema 文本：schema 文本是给人读的，
        改一次文案就会让反解失效，而校验悄无声息地不再生效是最难发现的那类退化。
        """
        return describe_tables(self.db_path) if self.db_path else {}

    @cached_property
    def df_columns(self) -> dict:
        """单表路线的列（表名固定叫 dataframe），与 db_columns 同形状，便于共用校验。"""
        return {"dataframe": [str(c) for c in self.df.columns]}

    @cached_property
    def data_profile_text(self) -> str:
        return format_data_profile(self.data_profile)

    @cached_property
    def enums(self) -> list:
        """枚举列的真实取值（取自数据画像），供取值校验与范围守卫共用。"""
        return list(self.data_profile.get("enums") or [])

    @property
    def today(self):
        """相对时间的参考日。默认运行当天，`AGENT_TODAY` 可固定口径。"""
        from app.config import AGENT_TODAY
        from app.tools.time_anchor import to_date
        return to_date(AGENT_TODAY) if AGENT_TODAY else to_date(None)

    @property
    def date_coverage(self) -> dict:
        """主日期列的覆盖范围 {min, max}，供时间锚点判断是否无交集。"""
        for item in self.data_profile.get("date_ranges") or []:
            if str(item.get("column", "")).lower() in ("order_date", "date", "ordertime"):
                return {"min": item.get("min"), "max": item.get("max")}
        ranges = self.data_profile.get("date_ranges") or []
        return dict(ranges[0]) if ranges else {}

    def time_anchors(self, question: str) -> list:
        """解析相对时间（`today` 与覆盖范围来自本上下文）。"""
        return resolve_time_anchor(question, today=self.today, coverage=self.date_coverage)

    @cached_property
    def catalog(self) -> dict:
        return load_catalog()

    @cached_property
    def lessons(self) -> list:
        """已沉淀的修正记录（P1-1），按当前库指纹隔离。

        只加载不挑选：挑哪几条要看具体问题，属于生成节点的职责
        （`select_lessons`）。这里做缓存是因为一次问答里可能重试多次，
        每次都读盘没必要。
        """
        if not self.use_lessons:
            return []
        return load_lessons(self)

    @cached_property
    def terms_dict(self) -> dict:
        """业务术语表本身（不是格式化后的文本），供相关度判断区分"库内词/库外词"。"""
        return self.catalog.get("terms") or {}

    @cached_property
    def known_words(self) -> set:
        """库内已知的业务词（术语名 + 字段标签 + 枚举取值）。

        不在其中的实义词说明是用户自定义口径（如"大单"），只能靠修正记录传下去。
        """
        return lesson_known_words(self.enums, self.terms_dict)

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
