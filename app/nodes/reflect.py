"""反思与重试节点（对应功能 F3.3，阶段 3 实现）。

职责：执行失败时，把错误信息回喂给 LLM 请求修正代码，限次重试。
"""
from app.state import AgentState


def reflect(state: AgentState) -> AgentState:
    """错误反思（占位，阶段 3 实现）。"""
    # TODO: 把 generated_code + error 回喂 LLM，生成修正代码
    return state
