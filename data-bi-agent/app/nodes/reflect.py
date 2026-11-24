"""错误反思与路由节点（对应功能 F3.3 / S3.4，阶段 3 实现）。

两条策略：
1. **换路优先于死磕**：SQL 失败后不硬重试同一路径，而是切到 pandas。
   原因：同一份 schema 下 SQL 出错常是字段名/方言理解的系统性偏差，重问同类问题
   往往还是错；换成单表宽表路径，等价信息用另一种表达方式问，成功率明显更高。
2. **错误回喂**：每次重试都把上一次的真实报错带进 Prompt（由 generate 节点处理），
   避免模型重复同一个错误。

重试次数由 state.max_retries 控制（默认 2）。
"""
from app.context import AgentContext
from app.state import AgentState

DEFAULT_MAX_RETRIES = 2
FALLBACK_ROUTE = "pandas"


def make_reflect_node(ctx: AgentContext, fallback_route: str = FALLBACK_ROUTE):
    """构造反思节点：累计重试次数、记录错误、必要时切换执行路径。"""

    def node(state: AgentState) -> dict:
        retry_count = state.get("retry_count", 0) + 1
        errors = list(state.get("errors") or [])
        if state.get("error"):
            errors.append(state["error"])

        current_route = state.get("route", "sql")
        next_route = current_route
        if current_route == "sql" and ctx.has_db:
            next_route = fallback_route

        update = {
            "retry_count": retry_count,
            "errors": errors,
            "route": next_route,
            "statement": "",
            # 注意：不清空 error —— 下一轮 generate 要把它作为修正提示回喂给模型
        }
        if next_route != current_route:
            update["schema_info"] = ctx.db_schema if next_route == "sql" else ctx.df_schema
        return update

    return node


def route_after_execute(state: AgentState) -> str:
    """条件边：执行后决定去回答还是去反思重试。"""
    if state.get("result") is not None:
        return "answer"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
    return "reflect" if retry_count < max_retries else "answer"
