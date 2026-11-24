"""回答生成节点（对应功能 F2.5，阶段 2 实现）。

职责：基于分析结果生成中文洞察文案。
"""
from app.state import AgentState


def generate_insight(state: AgentState) -> AgentState:
    """生成洞察文案（占位，阶段 2 实现）。"""
    # TODO: 把结果摘要喂给 LLM，产出含关键数字的结论
    return state
