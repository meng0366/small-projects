"""可视化工具（对应功能 F2.5，阶段 2 实现）。

选图策略：**规则优先，不调 LLM**。
理由：图表类型由结果集的列结构决定，规则判定准确且零成本；
让 LLM 选图反而增加延迟与不确定性（PandasAI 早期版本也曾用 LLM 选图，后改为规则 + 少量模型判断）。

支持的图表：
- 时间序列（时间列 + 数值列）→ 折线图
- 类别对比（类别列 + 数值列）→ 柱状图（类别名较长时自动转横向）
- 占比（列名含占比/份额/比例 且类别 ≤ 8）→ 饼图
- 双数值 → 散点图
- 单数值列 → 直方图
- 单行单列 / 无法判定 → 不画图（直接看数字更清楚）

输出为独立 HTML（plotly.js 内嵌，离线可打开），无需 kaleido。
"""
import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.config import DATA_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), "output")

# 判定为"时间维度"的列名关键词
TIME_HINTS = ["月", "季度", "年", "日", "date", "month", "year", "day", "period", "week"]
# 判定为"占比"的列名关键词
SHARE_HINTS = ["占比", "份额", "比例", "百分比", "share", "ratio", "pct"]

# 类别过多时不画图，避免图糊成一团
MAX_CATEGORIES = 20
MAX_ROWS_FOR_CHART = 200


def classify_columns(df: pd.DataFrame) -> dict:
    """把列分为时间列 / 数值列 / 类别列。"""
    time_cols, numeric_cols, category_cols = [], [], []

    for col in df.columns:
        s = df[col]
        name = str(col).lower()

        if pd.api.types.is_datetime64_any_dtype(s):
            time_cols.append(col)
            continue

        # 形如 2025-03 / 2025-03-15 的字符串视为时间
        # 注意：pandas 3 起字符串列是 str dtype 而非 object，两种都要判
        is_text = pd.api.types.is_string_dtype(s) or s.dtype == object
        if is_text and len(s) > 0:
            non_null = s.dropna()
            sample = str(non_null.iloc[0]) if not non_null.empty else ""
            if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", sample) or any(h in name for h in TIME_HINTS):
                time_cols.append(col)
                continue

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            category_cols.append(col)

    return {"time": time_cols, "numeric": numeric_cols, "category": category_cols}


def detect_chart_type(df: pd.DataFrame) -> str:
    """规则式判断最合适的图表类型。"""
    if df is None or df.empty or df.shape[0] < 2:
        return "none"

    cols = classify_columns(df)
    n_time, n_num, n_cat = len(cols["time"]), len(cols["numeric"]), len(cols["category"])

    if n_time >= 1 and n_num >= 1:
        return "line"
    if n_cat >= 1 and n_num >= 1:
        if n_cat == 1 and n_num == 1 and df.shape[0] <= 8:
            name = str(cols["numeric"][0]).lower()
            if any(h in name for h in SHARE_HINTS):
                return "pie"
        return "bar"
    if n_num >= 2:
        return "scatter"
    if n_num == 1:
        return "histogram"
    return "none"


def _safe_series(df: pd.DataFrame, col) -> pd.Series:
    """把列转成可比较的数值或字符串。"""
    return df[col]


def build_chart(df: pd.DataFrame, question: str = "", max_rows: int = MAX_ROWS_FOR_CHART):
    """按规则生成 Plotly 图。返回 (figure, chart_type)；不适用时返回 (None, "none")。"""
    chart_type = detect_chart_type(df)
    if chart_type == "none":
        return None, "none"

    data = df.head(max_rows).copy()
    cols = classify_columns(data)
    title = question or "查询结果"

    if chart_type == "line":
        x, y = cols["time"][0], cols["numeric"][0]
        data = data.sort_values(by=x)
        fig = px.line(data, x=x, y=y, markers=True, title=title)

    elif chart_type == "bar":
        x, y = cols["category"][0], cols["numeric"][0]
        data = data.sort_values(by=y, ascending=False).head(MAX_CATEGORIES)
        # 类别名较长（中文地名/品名）时用横向柱状，标签不易重叠
        long_labels = data[x].astype(str).str.len().mean() > 4 or data.shape[0] > 8
        if long_labels:
            fig = px.bar(data, x=y, y=x, orientation="h", title=title)
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        else:
            fig = px.bar(data, x=x, y=y, title=title)

    elif chart_type == "pie":
        names, values = cols["category"][0], cols["numeric"][0]
        fig = px.pie(data, names=names, values=values, title=title, hole=0.35)

    elif chart_type == "scatter":
        x, y = cols["numeric"][0], cols["numeric"][1]
        fig = px.scatter(data, x=x, y=y, title=title)

    elif chart_type == "histogram":
        x = cols["numeric"][0]
        fig = px.histogram(data, x=x, title=title)

    else:
        return None, "none"

    fig.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        font={"size": 13},
    )
    return fig, chart_type


def save_chart(fig: go.Figure, out_dir: str = None, prefix: str = "chart") -> str:
    """保存为独立 HTML（plotly.js 内嵌，双击即可离线查看）。"""
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.html"
    path = os.path.join(out_dir, filename)
    fig.write_html(path, include_plotlyjs=True)
    return path


CHART_TYPE_LABEL = {
    "line": "折线图（时间趋势）",
    "bar": "柱状图（类别对比）",
    "pie": "饼图（占比构成）",
    "scatter": "散点图（两指标关系）",
    "histogram": "直方图（数值分布）",
    "none": "不绘图",
}
