"""错误反思与路由节点（对应功能 F3.3 / S3.4，阶段 3 实现；空结果自愈为后续增强）。

三条策略：
1. **换路优先于死磕**：SQL 失败后不硬重试同一路径，而是切到 pandas。
   原因：同一份 schema 下 SQL 出错常是字段名/方言理解的系统性偏差，重问同类问题
   往往还是错；换成单表宽表路径，等价信息用另一种表达方式问，成功率明显更高。
2. **错误回喂**：每次重试都把上一次的真实报错带进 Prompt（由 generate 节点处理），
   避免模型重复同一个错误。
3. **空结果也触发一次自愈**（借鉴 ReFoRCE 的 simplify 思路）：
   `执行成功但返回 0 行`并不等于"问题答完了"，常见原因是筛选条件过严或枚举值写法
   与库中不一致。这种情况下**不换路**（换路解决的是能力问题，空结果多半是条件问题），
   只把"结果为空、请核对取值并放宽条件"回喂一次。
   只给一次机会：若放宽后仍为空，说明该条件下确实没有数据，如实告诉用户，
   而不是反复空转烧 token。

重试次数由 state.max_retries 控制（默认 2）。
"""
from app.context import AgentContext
from app.state import AgentState

DEFAULT_MAX_RETRIES = 2
FALLBACK_ROUTE = "pandas"

EMPTY_RESULT_NOTE = (
    "查询执行成功，但在当前条件下返回 0 行数据（不是语法错误）。"
    "常见原因：筛选条件过严或条件叠加后无交集、枚举值写法与库中不一致、"
    "时间范围超出数据覆盖区间。"
)


def is_empty_result(result) -> bool:
    """结果是否为"空但合法"（执行成功、0 行）。

    与失败区分：None 表示没跑出结果（错误），空 DataFrame 表示跑通了但没数据。
    """
    if result is None:
        return False
    empty_attr = getattr(result, "empty", None)
    if isinstance(empty_attr, bool):
        return empty_attr and len(result) == 0
    try:
        return len(result) == 0
    except TypeError:
        return False


def make_reflect_node(ctx: AgentContext, fallback_route: str = FALLBACK_ROUTE):
    """构造反思节点：累计重试次数、记录错误、必要时切换执行路径。"""

    def node(state: AgentState) -> dict:
        retry_count = state.get("retry_count", 0) + 1
        errors = list(state.get("errors") or [])
        empty = is_empty_result(state.get("result"))

        if empty:
            errors.append(EMPTY_RESULT_NOTE)
        elif state.get("error"):
            errors.append(state["error"])

        current_route = state.get("route", "sql")
        next_route = current_route

        # 空结果是"条件问题"不是"路径问题"：留在原路径重写，不换路。
        # 只有真正的报错才触发换路（SQL 出错 → pandas）。
        if not empty and current_route == "sql" and ctx.has_db:
            next_route = fallback_route

        update = {
            "retry_count": retry_count,
            "errors": errors,
            "route": next_route,
            "statement": "",
            # 注意：empty_result 保持不动 —— generate 靠它判断该用"放宽条件"
            # 还是"修语法"的反馈模板；下一轮 execute 会重新覆盖它。
            # 同理不清空 error —— 报错重试时 generate 要把它回喂给模型。
        }
        if empty:
            update["empty_retried"] = True
            update["error"] = ""
        if next_route != current_route:
            update["schema_info"] = ctx.db_schema if next_route == "sql" else ctx.df_schema
        return update

    return node


def route_after_execute(state: AgentState) -> str:
    """条件边：执行后决定去回答还是去反思重试。

    三种情况：
    - 有非空结果 → 直接回答；
    - 空结果且还没因空结果重试过 → 反思一次（放宽条件）；
    - 空结果但已经重试过 / 出错且重试次数用尽 → 回答（如实说明没数据或失败）。
    """
    result = state.get("result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if result is not None:
        if is_empty_result(result) and not state.get("empty_retried") \
                and retry_count < max_retries:
            return "reflect"
        return "answer"

    return "reflect" if retry_count < max_retries else "answer"
