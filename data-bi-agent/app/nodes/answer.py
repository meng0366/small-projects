"""回答生成节点（对应功能 F2.6，阶段 2 实现）。

两步走（控制 token 与成本，也避免把整表数据发给模型）：
1. 本地把结果压缩成"统计摘要"（行数、列、数值列的合计/极值、前若干行）；
2. 把摘要 + 原问题交给 LLM，产出 2-3 句含具体数字的中文结论。

失败降级：LLM 不可用或超时，回退到本地摘要生成的简单结论，保证流程不中断。

（阶段 3 会把它接入 LangGraph 状态机，届时用 generate_insight_from_state。）
"""
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.prompts.templates import INSIGHT_SYSTEM, INSIGHT_USER_TEMPLATE

MAX_SUMMARY_ROWS = 8


def summarize_result(result, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """把执行结果压缩成给 LLM 看的统计摘要（不发送整表）。"""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "结果：0 行（当前条件下没有匹配数据）"

        rows, cols = result.shape
        lines = [f"结果：{rows} 行 × {cols} 列", f"列名：{'、'.join(str(c) for c in result.columns)}"]

        for col in result.select_dtypes("number").columns[:3]:
            s = result[col].dropna()
            if s.empty:
                continue
            lines.append(
                f"- {col}：合计 {s.sum():,.2f}，最大 {s.max():,.2f}，"
                f"最小 {s.min():,.2f}，均值 {s.mean():,.2f}"
            )

        shown = result.head(max_rows)
        lines.append(f"前 {len(shown)} 行明细：")
        lines.append(shown.to_string(index=False))
        return "\n".join(lines)

    if isinstance(result, pd.Series):
        s = result.dropna()
        if s.empty:
            return "结果：空序列"
        return (f"结果：{len(s)} 项，合计 {s.sum():,.2f}，"
                f"最大 {s.max():,.2f}，最小 {s.min():,.2f}\n"
                + s.head(max_rows).to_string())

    return f"结果：{result}"


def local_insight(result) -> str:
    """不依赖 LLM 的兜底结论。"""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "未查询到符合条件的数据。"
        num_cols = list(result.select_dtypes("number").columns)
        if num_cols:
            col = num_cols[0]
            idx_max = result[col].idxmax()
            label = str(result.iloc[idx_max, 0]) if result.shape[1] > 1 else "最大值"
            return (f"共返回 {len(result)} 行结果，{col} 合计 "
                    f"{result[col].sum():,.2f}，其中 {label} 最高，为 {result[col].max():,.2f}。")
        return f"共返回 {len(result)} 行结果。"
    return f"结果为：{result}"


def generate_insight(question: str, result) -> str:
    """生成中文洞察文案；失败时降级为本地结论。"""
    summary = summarize_result(result)

    try:
        resp = get_llm().invoke([
            SystemMessage(content=INSIGHT_SYSTEM),
            HumanMessage(content=INSIGHT_USER_TEMPLATE.format(
                question=question, summary=summary)),
        ])
        text = (resp.content or "").strip()
        return text or local_insight(result)
    except Exception:
        return local_insight(result)


def generate_insight_from_state(state: dict) -> dict:
    """LangGraph 节点入口（阶段 3 使用）：写入 state['insight']。"""
    state["insight"] = generate_insight(state.get("question", ""), state.get("result"))
    return state
