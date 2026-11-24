"""LLM 客户端与输出清洗（对应功能 F0.1）。

单独成模块，供 code 生成、SQL 生成、洞察生成共用，避免互相 import。
"""
import re

from langchain_openai import ChatOpenAI

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def get_llm() -> ChatOpenAI:
    """构造 LLM 客户端（temperature=0 保证输出稳定可复现）。"""
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )


def strip_code_fence(text: str) -> str:
    """去掉模型可能输出的 markdown 代码块标记（支持 ```python / ```sql / 裸 ```）。"""
    text = text.strip()
    fence = re.search(r"```[a-zA-Z]*\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text
