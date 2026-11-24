"""问题理解节点（对应功能 F3.2，阶段 3 实现）。

职责：解析用户问题，结合 schema 与对话历史，生成结构化的分析意图。
"""
from app.state import AgentState


def understand(state: AgentState) -> AgentState:
    """理解用户问题（占位，待实现）。"""
    # TODO: 结合对话历史改写问题，补全指代（如"只看华东"）
    return state
