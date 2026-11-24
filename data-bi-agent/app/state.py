"""Agent 状态结构定义（对应功能 F3.1，阶段 3 完善）。

设计原则：
- 只放**可序列化**的数据（字符串、数字、列表、DataFrame 由 LangGraph 进程内传递）；
- 运行时重资源（数据库路径、宽表、schema）放在 AgentContext（app/context.py），不进 state；
- 所有字段 total=False，节点只更新自己负责的键，返回 dict 由 LangGraph 合并。
"""
from typing import Any, List, TypedDict


class Message(TypedDict):
    """一轮对话记录。"""
    role: str      # "user" | "assistant"
    content: str


class AgentState(TypedDict, total=False):
    """一次分析会话的完整状态。"""
    # ---- 输入 ----
    question: str                    # 用户当前提问（原文）
    history: List[Message]           # 历史对话（多轮记忆）

    # ---- 理解阶段 ----
    rewritten_question: str          # 结合上下文改写后的完整问题
    schema_info: str                 # 当前路径对应的表结构描述
    terms: str                       # 业务术语口径
    route: str                       # "sql" | "pandas"
    unknown_enum: Any                # 用户提到的枚举值库里没有时，记录澄清信息（P0-1b）

    # ---- 生成与执行 ----
    statement: str                   # 生成的 SQL 或 Python 代码
    statements: List[str]            # 历次生成的语句（判断"重试是否真的改过"，P0-1b）
    result: Any                      # 执行结果（DataFrame）

    # ---- 错误与重试 ----
    error: str                       # 最近一次错误
    errors: List[str]                # 历史错误（供反思与排查）
    retry_count: int                 # 已重试次数
    max_retries: int                 # 最大重试次数
    empty_result: bool               # 本次结果是否为空（0 行）——执行成功但没查到数据
    empty_retried: bool              # 是否已因空结果重试过（只给一次机会，避免空转）

    # ---- 输出 ----
    insight: str                     # 洞察文案
    chart_path: str                  # 图表 HTML 路径
    chart_type: str                  # 图表类型标识
    answer: str                      # 最终回答（失败时的说明文本）
    suggestions: List[str]           # 追问建议（规则式生成，阶段 6）
