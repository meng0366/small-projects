"""问题理解节点（对应功能 F3.1 / S3.2，阶段 3 实现）。

职责：
1. 多轮记忆：有历史时，把追问改写成不依赖上下文的完整问题（"只看华东" → "华东各品类销售额"）；
2. 取值校验：用户提到的枚举值库里没有时，标记出来交给澄清分支，别让模型硬映射（P0-1b）；
3. 路由判定：有数据库走 Text-to-SQL，否则走 Text-to-Pandas；
4. 注入对应的 schema 与业务术语。

省成本设计：**没有历史时不调用 LLM**，直接用原问题——单轮场景不该为改写付 token。
取值校验也纯规则，不额外调用模型。
"""
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.context import AgentContext
from app.llm import get_llm
from app.prompts.templates import REWRITE_SYSTEM, REWRITE_USER_TEMPLATE
from app.state import AgentState
from app.tools.attribution import detect_attribution
from app.tools.enum_guard import find_unknown_enum

HISTORY_LIMIT = 6  # 最多带入 6 条历史（约 3 轮）


def format_history(history: List[dict], limit: int = HISTORY_LIMIT) -> str:
    """把对话历史格式化为可读文本。"""
    if not history:
        return "（无）"
    recent = history[-limit:]
    lines = []
    for msg in recent:
        role = "用户" if msg.get("role") == "user" else "分析结果"
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def rewrite_question(question: str, history: List[dict]) -> str:
    """结合历史改写问题；无历史或调用失败时返回原问题。"""
    if not history:
        return question

    try:
        resp = get_llm().invoke([
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(content=REWRITE_USER_TEMPLATE.format(
                history=format_history(history), question=question)),
        ])
        text = (resp.content or "").strip().strip('"').strip("“”")
        return text or question
    except Exception:
        return question


def make_understand_node(ctx: AgentContext):
    """构造理解节点（闭包持有运行时上下文）。"""

    def node(state: AgentState) -> dict:
        question = state.get("question", "")
        history = state.get("history") or []

        rewritten = rewrite_question(question, history)
        route = "sql" if ctx.has_db else "pandas"
        schema_info = ctx.db_schema if route == "sql" else ctx.df_schema

        # 取值校验：拿**改写后**的问题去比对，多轮追问里的取值也能被拦住
        unknown = find_unknown_enum(rewritten, ctx.data_profile.get("enums") or [])

        # 归因意图（P1-2）：纯规则识别，不额外调用模型。
        # 只在 SQL 路线启用：归因需要一条"对比列 × 维度 × 指标"的交叉表，
        # 对应的提示词是 SQL 版；单表 pandas 路线没有等价模板，
        # 硬做只会拆出一张对不上的表，不如老实用普通问答。
        intent = detect_attribution(rewritten, ctx.enums) if route == "sql" else None

        return {
            "rewritten_question": rewritten,
            "route": route,
            "schema_info": schema_info,
            "terms": ctx.terms,
            "retry_count": state.get("retry_count", 0),
            "unknown_enum": unknown,
            "attribution_intent": intent,
        }

    return node
