"""数据加载与 Schema 描述生成。

对应功能：
- F1.1 CSV/Excel 数据加载
- F1.2 SQLite 多表接入
- F1.3 Schema 描述生成（DataFrame 版 / 数据库版）
- F1.4 业务术语词典读取（借鉴 OpenChatBI 的"Schema 是元数据"设计）

参考项目说明：
- OpenChatBI（zhongyu09/openchatbi）把表结构、字段释义、指标口径作为元数据统一管理，
  并在生成 SQL 时做 schema 关联。本项目用 catalog.yaml 承担同样的角色。
- WrenAI 的"上下文层"：把枚举值、单位、join 关系显式给出，显著降低模型臆造字段的概率。
"""
import os
import sqlite3

import pandas as pd
import yaml

from app.config import DATA_DIR

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.yaml")

# 默认数据库（模拟销售库）
DEFAULT_DB = "sales.db"


def load_dataframe(file_name: str = "sales_orders.csv") -> pd.DataFrame:
    """加载 data/ 目录下的数据文件（CSV 或 Excel）。"""
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"数据文件不存在：{path}\n"
            f"请先运行 `python data/generate_mock_data.py` 生成示例数据，"
            f"或把自备数据放到 data/ 目录。"
        )

    ext = os.path.splitext(file_name)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def explore_dataframe(df: pd.DataFrame) -> str:
    """打印数据结构概览，返回概览文本（S1.2）。"""
    lines = [
        f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列",
        f"字段列表：{list(df.columns)}",
        "",
        "字段类型：",
        df.dtypes.to_string(),
        "",
        "前 3 行样例：",
        df.head(3).to_string(index=False),
    ]
    return "\n".join(lines)


def build_schema_text(df: pd.DataFrame, max_examples: int = 3) -> str:
    """生成供 Prompt 使用的 schema 文本（S1.3）。

    包含：表规模、每个字段的名称/类型/样例值。
    """
    lines = [f"数据表共 {df.shape[0]} 行，字段如下："]
    for col in df.columns:
        dtype = str(df[col].dtype)
        # 取非空样例值
        samples = df[col].dropna().unique()[:max_examples]
        sample_text = "、".join(str(s) for s in samples)
        lines.append(f"- {col}（{dtype}）：示例值 {sample_text}")
    return "\n".join(lines)


def load_catalog_terms(path: str = CATALOG_PATH) -> str:
    """读取业务术语词典，格式化为 Prompt 可用的文本（F1.4）。"""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f) or {}

    terms = catalog.get("terms") or {}
    if not terms:
        return ""
    lines = ["业务术语口径："]
    for name, definition in terms.items():
        lines.append(f"- {name}：{definition}")
    return "\n".join(lines)


def load_catalog(path: str = CATALOG_PATH) -> dict:
    """读取完整语义词典（表/字段释义、关系、术语）。"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============ 数据库侧（阶段 2 / F1.2、F1.3） ============

def get_db_path(db_name: str = DEFAULT_DB) -> str:
    """拼接 data/ 下的数据库路径并校验存在性。"""
    path = os.path.join(DATA_DIR, db_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"数据库文件不存在：{path}\n"
            f"请先运行 `python data/generate_mock_data.py` 生成示例数据。"
        )
    return path


def list_tables(db_path: str) -> list:
    """列出数据库中的用户表（排除 sqlite 内部表）。"""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def _table_columns(db_path: str, table: str) -> list:
    """读取某张表的真实列信息：[(name, type), ...]。"""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        return [(r[1], r[2] or "TEXT") for r in rows]
    finally:
        con.close()


def _table_row_count(db_path: str, table: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def build_db_schema_text(db_path: str, catalog: dict = None) -> str:
    """生成供 Text-to-SQL 使用的 Schema 文本（F1.2 + F1.3）。

    与 DataFrame 版的区别：额外给出表关系、枚举值、行数——这正是 WrenAI
    "上下文层" 的做法，能显著减少多表 JOIN 与字段取值的臆造。

    注意：真实列类型以 PRAGMA 为准，catalog.yaml 提供中文释义与示例值。
    """
    catalog = catalog if catalog is not None else load_catalog()
    catalog_tables = {t.get("name"): t for t in (catalog.get("tables") or [])}

    tables = list_tables(db_path)
    lines = [f"数据库：{os.path.basename(db_path)}（SQLite），共 {len(tables)} 张表", ""]

    for table in tables:
        meta = catalog_tables.get(table, {})
        label = meta.get("label", "")
        count = _table_row_count(db_path, table)
        header = f"表 {table}"
        if label:
            header += f"（{label}）"
        header += f"，{count} 行"
        lines.append(header)

        if meta.get("description"):
            lines.append(f"  说明：{meta['description']}")

        col_meta = meta.get("columns") or {}
        for col_name, col_type in _table_columns(db_path, table):
            m = col_meta.get(col_name, {})
            parts = [f"  - {col_name} {col_type}"]
            if m.get("label"):
                parts.append(m["label"])
            if m.get("example") is not None:
                parts.append(f"示例 {m['example']}")
            if m.get("note"):
                parts.append(m["note"])
            if m.get("ref"):
                parts.append(f"外键 → {m['ref']}")
            lines.append("  ".join(parts))
        lines.append("")

    relationships = catalog.get("relationships") or []
    if relationships:
        lines.append("表关系：")
        lines.extend(f"- {r}" for r in relationships)
        lines.append("")

    # 枚举值：SQL 里写死字面量容易出错，单独列出
    enum_lines = []
    for table, meta in catalog_tables.items():
        for col_name, m in (meta.get("columns") or {}).items():
            if m.get("values"):
                enum_lines.append(f"- {table}.{col_name}：{' / '.join(m['values'])}")
    if enum_lines:
        lines.append("枚举字段取值范围（写 WHERE 时须严格使用以下字面量）：")
        lines.extend(enum_lines)
        lines.append("")

    lines.append("数据库方言：SQLite。日期字段 order_date 以 'YYYY-MM-DD' 文本形式存储，"
                 "按月分组用 strftime('%Y-%m', order_date)，按年用 strftime('%Y', order_date)。")
    return "\n".join(lines).strip()
