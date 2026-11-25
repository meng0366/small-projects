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


def describe_tables(db_path: str) -> dict:
    """`{表名: [列名]}`：给"计划校验"用，判断计划里提到的表和列是否真实存在。

    为什么不解析 schema 文本：`build_db_schema_text` 是给人/模型读的 markdown，
    格式会随文案调整，拿它反解表列会在某次改文案后悄悄失效。
    直接读库结构，改文案不会影响校验。
    """
    if not db_path or not os.path.exists(db_path):
        return {}
    try:
        return {table: [name for name, _ in _table_columns(db_path, table)]
                for table in list_tables(db_path)}
    except sqlite3.Error:
        return {}


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

    # 枚举值 + 时间范围用**实测**值（catalog 声明可能已过期）：
    # 真实试跑里模型会把"办公用品"映射成"配件"，说明只给列表还不够，
    # 但至少要让列表是库里的真值，并把"哪些时段根本没有数据"明确摆出来。
    profile = collect_data_profile(db_path, catalog)
    if profile["enums"]:
        lines.append("枚举字段取值范围（写 WHERE 时必须严格使用以下字面量，不要用近义词替换）：")
        for e in profile["enums"]:
            suffix = "（仅列前 20 个）" if len(e["values"]) >= MAX_ENUM_VALUES else ""
            lines.append(f"- {e['table']}.{e['column']}：{' / '.join(e['values'])}{suffix}")
        lines.append("")
    if profile["date_ranges"]:
        lines.append("时间覆盖范围（超出该范围的时段确实没有数据，不要用别的时段顶替）：")
        for d in profile["date_ranges"]:
            lines.append(f"- {d['table']}.{d['column']}：{d['min']} ~ {d['max']}")
        lines.append("")

    lines.append("数据库方言：SQLite。日期字段 order_date 以 'YYYY-MM-DD' 文本形式存储，"
                 "按月分组用 strftime('%Y-%m', order_date)，按年用 strftime('%Y', order_date)。")
    return "\n".join(lines).strip()


# ============ 数据画像（P0-1b：空结果判断 + 取值核对） ============

MAX_ENUM_VALUES = 20
"""单列最多列出的取值个数，防止高基数列（如客户名）把 Prompt 撑爆。"""


def _distinct_values(con, table: str, column: str, limit: int = MAX_ENUM_VALUES) -> list:
    """读某列的真实取值（去重、排序、限量）。失败返回空列表，让调用方回退。"""
    try:
        rows = con.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL ORDER BY "{column}" LIMIT {int(limit)}'
        ).fetchall()
    except sqlite3.Error:
        return []
    return [str(r[0]) for r in rows]


def collect_data_profile(db_path: str, catalog: dict = None) -> dict:
    """实测库里"实际有什么"：日期覆盖范围 + 枚举列真实取值。

    为什么不复用 schema：schema 描述的是"表结构长什么样"，而回答"为什么查不到"
    需要的是事实——真实时间范围、真实取值。catalog 里声明的 values 也可能与库不符。
    没有这层事实，模型只能猜"是不是条件写错了"，或者把用户说的词映射成近似取值。

    返回结构：
        {"date_ranges": [{table, column, min, max}],
         "enums": [{table, column, label, values, hints}]}
    """
    catalog = catalog if catalog is not None else load_catalog()
    catalog_tables = {t.get("name"): t for t in (catalog.get("tables") or [])}
    profile = {"date_ranges": [], "enums": []}
    if not db_path or not os.path.exists(db_path):
        return profile

    con = sqlite3.connect(db_path)
    try:
        for table in list_tables(db_path):
            for col_name, col_type in _table_columns(db_path, table):
                if "date" not in (col_type or "").lower():
                    continue
                try:
                    mn, mx = con.execute(
                        f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM "{table}"'
                    ).fetchone()
                except sqlite3.Error:
                    continue
                if mn and mx:
                    profile["date_ranges"].append({
                        "table": table, "column": col_name,
                        "min": str(mn), "max": str(mx),
                    })

            col_meta = (catalog_tables.get(table) or {}).get("columns") or {}
            for col_name in sorted({c for c, m in col_meta.items() if m.get("values")}):
                m = col_meta.get(col_name) or {}
                # 以实测为准，实测读不到（列不存在等）才回退到 catalog 声明
                values = _distinct_values(con, table, col_name) or [
                    str(v) for v in (m.get("values") or [])]
                if not values:
                    continue
                profile["enums"].append({
                    "table": table, "column": col_name,
                    "label": m.get("label", ""),
                    "values": values,
                    "hints": [str(h) for h in (m.get("hints") or [])],
                })
    finally:
        con.close()
    return profile


def format_data_profile(profile: dict) -> str:
    """把数据画像排成 Prompt 可用文本；没有可用事实时返回空串。"""
    date_ranges = (profile or {}).get("date_ranges") or []
    enums = (profile or {}).get("enums") or []
    if not date_ranges and not enums:
        return ""

    lines = []
    if date_ranges:
        lines.append("时间覆盖范围：")
        lines.extend(
            f"- {d['table']}.{d['column']}：{d['min']} ~ {d['max']}" for d in date_ranges)
    if enums:
        if lines:
            lines.append("")
        lines.append("字段实际取值（提问里出现的值必须来自这里）：")
        for e in enums:
            suffix = "（仅前 20 个）" if len(e["values"]) >= MAX_ENUM_VALUES else ""
            lines.append(f"- {e['table']}.{e['column']}：{' / '.join(e['values'])}{suffix}")
    return "\n".join(lines)
