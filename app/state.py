"""Agent 状态结构定义（对应功能 F3.1，阶段 3 完善）。

这里先定义好数据结构，供后续 LangGraph 节点使用。
"""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """一次分析会话的完整状态。

    total=False 表示所有字段均为可选，方便流程中逐步填充。
    """
    question: str                    # 用户当前提问
    schema_info: str                 # 数据表结构描述
    generated_code: str              # 生成的代码/SQL
    result: Any                      # 执行结果（DataFrame 或结构化数据）
    chart: Any                       # 生成的图表对象
    insight: str                     # 洞察文案
    error: str                       # 最近一次错误信息
    retry_count: int                 # 当前重试次数
    messages: list                   # 对话历史（多轮记忆）
    answer: str                      # 最终回答
