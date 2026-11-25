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
    trace_id: str                    # 本次问答的追踪号（P2-2），把所有阶段日志串成一条链

    # ---- 理解阶段 ----
    rewritten_question: str          # 结合上下文改写后的完整问题
    schema_info: str                 # 当前路径对应的表结构描述
    terms: str                       # 业务术语口径
    route: str                       # "sql" | "pandas"
    unknown_enum: Any                # 用户提到的枚举值库里没有时，记录澄清信息（P0-1b）
    time_defer: str                  # 相对时间完全超出数据覆盖范围时的说明（P0）；非空则直接拒答
    attribution_intent: Any          # 归因意图：谁跟谁比（P1-2），None 表示普通问答
    plan_needed: bool                # 是否需要先出计划（P1）；决定 understand 之后走哪条边

    # ---- 计划（P1 计划先行）----
    plan: Any                        # 结构化查询计划（指标/表/维度/过滤/时间/假设）
    plan_hint: str                   # 计划拼成的生成引导语，注入生成阶段
    plan_issues: List[dict]          # 校验结果：fatal 会短路到澄清，warn 只作提示
    plan_used: bool                  # 本次是否真的用上了计划（解析失败时为 False）
    plan_reason: str                 # 用/不用的原因，便于排查"为什么这次没出计划"

    # ---- 生成与执行 ----
    statement: str                   # 生成的 SQL 或 Python 代码
    statements: List[str]            # 历次生成的语句（判断"重试是否真的改过"，P0-1b）
    lessons_used: List[str]          # 本次注入了哪几条修正记录（P1-1，便于排查"为什么这次答对了"）
    result: Any                      # 执行结果（DataFrame）

    # ---- 错误与重试 ----
    error: str                       # 最近一次错误
    errors: List[str]                # 历史错误（供反思与排查）
    retry_count: int                 # 已重试次数
    max_retries: int                 # 最大重试次数
    empty_result: bool               # 本次结果是否为空（0 行）——执行成功但没查到数据
    empty_retried: bool              # 是否已因空结果重试过（只给一次机会，避免空转）
    defer: str                       # 答不了的原因（模型声明缺字段 / 重试改动了用户条件，P0-2b）
    scope_violation: List[str]       # 重试时被改动的用户条件明细（展示给用户，P0-2b）

    # ---- 输出 ----
    insight: str                     # 洞察文案
    attribution: Any                 # 归因结果（含分项表与结论，P1-2）
    chart_path: str                  # 图表 HTML 路径
    chart_type: str                  # 图表类型标识
    answer: str                      # 最终回答（失败时的说明文本）
    suggestions: List[str]           # 追问建议（规则式生成，阶段 6）
