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
4. **聚合型空结果不重试**（P0-1b，真实试跑后补）：
   `SUM/AVG/MAX` 在没有匹配行时不返回 0 行，而是**返回一行 NULL**，
   所以 `df.empty` 是 False，只看行数会把"其实没数据"当成有结果。
   而这类问题一旦去"放宽条件"，WHERE 一松就会把别的时段、别的地区的数
   算进同一个指标 —— 用户看到的仍是一个像样的金额，错得毫无痕迹。
   因此聚合空一律如实说明、不重试（`COUNT(*)` 返回 0 属有效答案，不算空）。

重试次数由 state.max_retries 控制（默认 2）。
"""
import pandas as pd

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


def is_aggregate_empty(result) -> bool:
    """聚合查询的"无数据"是另一个样子：1 行 1 列，但值是 NULL。

    `SUM/AVG/MAX` 在没有匹配行时不返回 0 行，而返回一行 NULL，
    所以 `df.empty` 是 False —— 只按行数判断会把"其实没数据"当成有结果。
    真实试跑里"2024 年总销售额"就是这样漏判的：LLM 对着 None 编出
    "数据缺失或未成功汇总，需核查数据源"，把用户引向错误方向。

    只认 NULL/NaN，**不认 0**：`COUNT(*)` 无匹配返回 0，那是有效答案（确实 0 笔），
    `SUM` 有行但金额都是 0 也返回 0，同样不能说"没数据"。
    """
    if result is None or getattr(result, "shape", None) != (1, 1):
        return False
    try:
        value = result.iloc[0, 0]
    except Exception:
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
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
        # 但"复用的 SQL 过时"（cache_failed）也不换路：那是表结构/字段变了，
        # 换成 pandas 会绕开 JOIN 与聚合能力，正确的处置是重新生成一条 SQL。
        if not empty and current_route == "sql" and ctx.has_db \
                and not state.get("cache_failed"):
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

    五种情况：
    - 聚合型空（1 行 NULL，如 SUM 无匹配）→ 直接回答，**不重试**：
      放宽条件会把别的时段/地区的数算进同一个指标，产出"看起来正常"的错数；
    - 明细型空（0 行）且还没因空结果重试过 → 反思一次（放宽条件）；
    - 明细型空但已重试过 / 重试次数用尽 → 回答（如实说明没数据）；
    - 已判定答不了（defer：缺字段，或重试改动了用户条件）→ 直接回答，不再重试；
    - 真出错（result 为 None）→ 还有次数就反思（换路），否则回答失败说明。
    """
    result = state.get("result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    # 已经确定"这题答不了"：再让模型换路只会得到另一个问题或一个错数
    if state.get("defer"):
        return "answer"

    if result is not None:
        if is_aggregate_empty(result):
            return "answer"
        if is_empty_result(result) and not state.get("empty_retried") \
                and retry_count < max_retries:
            return "reflect"
        return "answer"

    return "reflect" if retry_count < max_retries else "answer"
