"""代码生成节点（对应功能 F2.1，阶段 1 实现）。

借鉴点：
- PandasAI 的 code_generation 模块：用 schema + few-shot 生成代码，且不发送整表数据；
- OpenChatBI 的 schema 关联：把业务术语注入 Prompt，减少口径误解。
"""
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from app.prompts.templates import FEW_SHOT_EXAMPLES, SYSTEM_ANALYST, USER_TEMPLATE


def get_llm() -> ChatOpenAI:
    """构造 LLM 客户端（temperature=0 保证稳定）。"""
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )


def strip_code_fence(text: str) -> str:
    """去掉模型可能输出的 markdown 代码块标记。"""
    text = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def build_messages(schema_info: str, terms: str, question: str) -> list:
    """构造带 few-shot 的消息列表。"""
    messages = [SystemMessage(content=SYSTEM_ANALYST)]
    for q, code in FEW_SHOT_EXAMPLES:
        messages.append(HumanMessage(content=USER_TEMPLATE.format(
            schema=schema_info, terms=terms, question=q)))
        messages.append(AIMessage(content=code))
    messages.append(HumanMessage(content=USER_TEMPLATE.format(
        schema=schema_info, terms=terms, question=question)))
    return messages


def generate_code(schema_info: str, terms: str, question: str) -> str:
    """根据问题 + schema + 术语生成 pandas 代码。"""
    llm = get_llm()
    messages = build_messages(schema_info, terms, question)
    resp = llm.invoke(messages)
    return strip_code_fence(resp.content)
