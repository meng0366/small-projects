"""数据加载与 Schema 描述生成。

对应功能：
- F1.1 CSV/Excel 数据加载
- F1.3 Schema 描述生成
- F1.4 业务术语词典读取（借鉴 OpenChatBI 的"Schema 是元数据"设计）

参考项目说明：
- OpenChatBI（zhongyu09/openchatbi）把表结构、字段释义、指标口径作为元数据统一管理，
  并在生成 SQL 时做 schema 关联。本项目用 catalog.yaml 承担同样的角色。
"""
import os

import pandas as pd
import yaml

from app.config import DATA_DIR

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.yaml")


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
