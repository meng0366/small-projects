"""结果渲染与导出（阶段 6：命令行增强）。

为什么单独一个模块：
- 命令行要支持 `table / markdown / csv / json` 四种输出，还要能落盘成
  csv / json / md / xlsx；这些是纯格式转换，和 Agent 业务无关；
- 网页端、命令行、评测都用到"把 DataFrame 变成可读文本"，集中一处避免三份实现漂移；
- 全部为纯函数（除了 export 写文件），方便离线测试。

设计取舍：
- **不引入 tabulate / rich 等新依赖**：markdown 表格手写十几行就够，
  依赖越少，别人 clone 下来越容易跑起来；
- **数值保留原样**：不在渲染层做四舍五入，避免"看到的数"和"算出的数"不一致。
"""
import json
import os
from typing import Any, Optional

import pandas as pd

SUPPORTED_FORMATS = ("table", "markdown", "csv", "json")

# 实际写盘支持的扩展名；xlsx 依赖 openpyxl（已在 requirements.txt 中）
EXPORT_SUFFIXES = {
    ".csv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".xlsx": "xlsx",
}


class UnsupportedFormat(ValueError):
    """未知的输出格式或导出扩展名。"""


def normalize_format(name: Optional[str]) -> str:
    """把用户输入规范化成标准格式名，非法值直接报错（而不是静默回退）。"""
    if not name:
        return "table"
    key = name.strip().lower()
    key = {"md": "markdown", "tbl": "table"}.get(key, key)
    if key not in SUPPORTED_FORMATS:
        raise UnsupportedFormat(
            f"不支持的输出格式：{name}（可选：{'、'.join(SUPPORTED_FORMATS)}）"
        )
    return key


def to_records(df: pd.DataFrame) -> list:
    """转成可 JSON 序列化的记录列表：NaN → null，日期 → ISO 字符串。"""
    if df is None:
        return []
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        return [{"value": df}]

    records = []
    for row in df.itertuples(index=False, name=None):
        item = {}
        for col, value in zip(df.columns, row):
            item[str(col)] = _clean_value(value)
        records.append(item)
    return records


def _clean_value(value: Any) -> Any:
    """把 pandas/numpy 的标量转成纯 Python 值，否则 json.dumps 会报错。"""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat(sep=" ")
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    return value


def to_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """对齐的纯文本表格（终端默认输出）。"""
    if df is None:
        return ""
    if isinstance(df, pd.Series):
        return df.to_string()
    if not isinstance(df, pd.DataFrame):
        return str(df)
    shown = df.head(max_rows)
    text = shown.to_string(index=False)
    if len(df) > max_rows:
        text += f"\n（共 {len(df)} 行，仅显示前 {max_rows} 行）"
    return text


def to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    """GitHub 风格 markdown 表格，可直接粘进 PR / 文档。"""
    if df is None:
        return ""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        return str(df)

    shown = df.head(max_rows)
    headers = [str(c).replace("|", "\\|") for c in shown.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in shown.itertuples(index=False, name=None):
        cells = []
        for value in row:
            text = "" if value is None else str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append("")
        lines.append(f"（共 {len(df)} 行，仅显示前 {max_rows} 行）")
    return "\n".join(lines)


def to_csv(df: pd.DataFrame) -> str:
    """CSV 文本（不带索引，便于 `|` 管道与 Excel 打开）。"""
    if df is None:
        return ""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        return str(df)
    return df.to_csv(index=False)


def to_json(df: pd.DataFrame, indent: int = 2) -> str:
    """记录数组形式的 JSON（脚本里最好解析的形态）。"""
    return json.dumps(to_records(df), ensure_ascii=False, indent=indent, default=str)


def render(df: pd.DataFrame, fmt: str = "table", max_rows: int = 50) -> str:
    """按格式渲染结果集。"""
    fmt = normalize_format(fmt)
    if fmt == "table":
        return to_table(df, max_rows)
    if fmt == "markdown":
        return to_markdown(df, max_rows)
    if fmt == "csv":
        return to_csv(df)
    return to_json(df)


def export(df: pd.DataFrame, path: str) -> str:
    """把结果集写到文件，格式由扩展名决定，返回绝对路径。

    CSV 用 utf-8-sig：Windows 上 Excel 双击打开不乱码（这是最常见的翻车点）。
    """
    if not path:
        raise UnsupportedFormat("导出路径不能为空")

    suffix = os.path.splitext(path)[1].lower()
    kind = EXPORT_SUFFIXES.get(suffix)
    if not kind:
        raise UnsupportedFormat(
            f"不支持的导出格式：{suffix or '（无扩展名）'}"
            f"（可选：{'、'.join(sorted(EXPORT_SUFFIXES))}）"
        )

    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df is None:
        df = pd.DataFrame()

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    if kind == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif kind == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_records(df), f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
    elif kind == "markdown":
        with open(path, "w", encoding="utf-8") as f:
            f.write(to_markdown(df, max_rows=len(df) or 1) + "\n")
    else:  # xlsx
        df.to_excel(path, index=False)

    return os.path.abspath(path)
