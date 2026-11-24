"""生成节点：pandas 代码（阶段 1）与 SQL 语句（阶段 2）。

对应功能：
- F2.1 Text-to-Pandas 代码生成
- F2.3 Text-to-SQL 生成

借鉴点：
- PandasAI 的 code_generation 模块：用 schema + few-shot 生成代码，且不发送整表数据；
- OpenChatBI 的 schema 关联：把业务术语注入 Prompt，减少口径误解。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import get_llm, strip_code_fence
from app.prompts.templates import (
    ERROR_FEEDBACK_TEMPLATE,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_SQL,
    SQL_USER_TEMPLATE,
    SYSTEM_ANALYST,
    SYSTEM_SQL,
    USER_TEMPLATE,
)


def _build_messages(system: str, examples: list, user_template: str,
                    schema_info: str, terms: str, question: str,
                    previous_error: str = None, kind: str = "SQL") -> list:
    """构造带 few-shot 的消息列表；重试时追加错误反馈消息。"""
    messages = [SystemMessage(content=system)]
    for q, output in examples:
        messages.append(HumanMessage(
            content=user_template.format(schema=schema_info, terms=terms, question=q)))
        messages.append(AIMessage(content=output))
    messages.append(HumanMessage(
        content=user_template.format(schema=schema_info, terms=terms, question=question)))

    if previous_error:
        messages.append(HumanMessage(content=ERROR_FEEDBACK_TEMPLATE.format(
            kind=kind, error=previous_error)))
    return messages


def generate_code(schema_info: str, terms: str, question: str,
                  previous_error: str = None) -> str:
    """根据问题 + schema + 术语生成 pandas 代码。"""
    messages = _build_messages(SYSTEM_ANALYST, FEW_SHOT_EXAMPLES, USER_TEMPLATE,
                               schema_info, terms, question,
                               previous_error, kind="Python 代码")
    resp = get_llm().invoke(messages)
    return strip_code_fence(resp.content)


def generate_sql(schema_info: str, terms: str, question: str,
                 previous_error: str = None) -> str:
    """根据问题 + 数据库 schema + 术语生成 SQL 语句（F2.3）。"""
    messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                               schema_info, terms, question,
                               previous_error, kind="SQL")
    resp = get_llm().invoke(messages)
    return strip_code_fence(resp.content)
