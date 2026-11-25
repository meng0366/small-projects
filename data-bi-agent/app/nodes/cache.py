"""SQL 复用节点（P1，2026-11-26）。

位置刻意放在 `understand` **之前**：复用要省的就是"让模型重写一遍 SQL"这一步，
如果先走理解再查缓存，虽然理解本身在单轮场景不调模型，但计划（P1）可能先花掉
一次调用，省下的钱又花回去了。

命中后要把 `understand` 本会填的字段补齐（route / schema / terms），
否则下游 execute 与 answer 会拿到一份缺字段的 state —— 这种"少一个键"
的错误不会报异常，只会让 reflect 之后的重试莫名走了错的分支。

不命中的原因（`cache_note`）会进 trace，排查"为什么这次没复用"时先看这里：
是问题变了、库变了、口径变了一样能一眼看出来。
"""
from app.context import AgentContext
from app.state import AgentState
from app.tools.sql_cache import find_cached_sql
from app.tools.time_anchor import format_time_hint


def make_cache_node(ctx: AgentContext):
    """构造缓存查询节点：命中则带着历史 SQL 直接去执行。"""

    def node(state: AgentState) -> dict:
        question = state.get("question", "")
        history = state.get("history") or []

        entry, reason = find_cached_sql(question, ctx, has_history=bool(history))
        if not entry:
            return {"cache_hit": False, "cache_note": reason}

        # 补齐理解阶段的产物。术语里含时间锚点提示 —— 命中时用不到，
        # 但如果这轮执行失败要重新生成，提示词得是完整的那一份。
        terms = ctx.terms
        try:
            hint = format_time_hint(ctx.time_anchors(question), today=ctx.today,
                                    coverage=ctx.date_coverage)
            if hint:
                terms = (terms + "\n\n" + hint).strip() if terms else hint
        except Exception:
            pass          # 锚点算不出来不影响执行已固化的 SQL

        return {
            "rewritten_question": question,
            "route": "sql",
            "schema_info": ctx.db_schema,
            "terms": terms,
            "statement": entry.get("sql", ""),
            "cache_hit": True,
            "cache_note": reason,
            "cached_insight": entry.get("insight") or "",
            "cached_hash": entry.get("result_fp") or "",
            "plan_needed": False,
            "plan_reason": "缓存命中",
            "attribution_intent": None,
            "unknown_enum": None,
            "time_defer": "",
        }

    return node
