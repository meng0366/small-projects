"""计划先行节点（P1，2026-11-25）。

一句话：**先让模型说清"要查什么"，代码校验通过后再让它写 SQL。**

为什么值得多花一次调用
----------------------
已有的守卫都在事后：SQL 报错才重试（reflect）、结果为空才放宽（P0-1）、
结果非空但条件被改掉才丢弃（scope_guard）。这些机制已经花掉一次调用、一条 SQL。
而"把问题理解错"这个错误在 SQL 写出来时就已经定型——语法对、结果合理、
数字和列名都正常，用户没有任何理由怀疑它。

计划先行把这一步提前：模型先输出指标 / 表 / 维度 / 过滤 / 时间 / 假设，
`plan_guard.validate_plan` 用代码核对三件事（时间有交集、取值真实存在、
年份没被换掉），通过才允许生成 SQL。

三条硬拦截都接到**已有的短路机制**上，不新造终止节点：
- 时间无交集 → `time_defer`（与相对时间锚点共用同一条说明路径）
- 取值不存在 → `unknown_enum`（与取值校验共用同一条澄清路径）
- 年份/月份被换 → `defer`（说明为什么不给数）

失败时的姿态
------------
- 计划**解析失败**：退回普通生成。计划是增强，不是门槛——不能因为它坏了就答不了问题；
- 计划**校验出 warn**（表名列名对不上）：只把提醒拼进生成提示，不阻断。
  列名可能带表前缀或别名，误拦的代价比漏拦大。
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.context import AgentContext
from app.llm import get_llm
from app.prompts.templates import PLAN_USER_TEMPLATE, SYSTEM_PLAN
from app.state import AgentState
from app.tools.plan_guard import (
    format_plan_hint,
    has_fatal,
    needs_plan,
    parse_plan,
    validate_plan,
)


def generate_plan(schema_info: str, terms: str, question: str) -> str:
    """让模型输出一份 JSON 计划（不写 SQL）。"""
    resp = get_llm().invoke([
        SystemMessage(content=SYSTEM_PLAN),
        HumanMessage(content=PLAN_USER_TEMPLATE.format(
            schema=schema_info, terms=terms, question=question)),
    ])
    return resp.content or ""


def build_plan(ctx: AgentContext, question: str, route: str = "sql",
               schema_info: str = "", terms: str = "", has_history: bool = False,
               intent: Optional[dict] = None, anchors: list = None) -> dict:
    """生成并校验计划，返回可直接并进 state 的更新。

    抽成函数而不是只写在节点里，是因为**线性编排也要用**：
    P2-2 那轮就吃过"只有状态机有链路、线性版没有"的亏，两个引擎必须走同一套判断。
    """
    update = {
        "plan": None, "plan_hint": "", "plan_issues": [],
        "plan_used": False, "plan_reason": "",
        "time_defer": "", "unknown_enum": None, "defer": "",
    }
    # 计划要多一次模型调用，必须能关（`--no-plan` / 网页开关）：
    # 用户可能只想省钱，也可能已经在别处把口径固定住了。
    if not getattr(ctx, "use_plan", True):
        update["plan_reason"] = "已关闭（--no-plan）"
        return update

    needed, reason = needs_plan(question, has_history=has_history,
                                anchors=anchors, intent=intent, route=route)
    update["plan_reason"] = reason
    if not needed:
        return update

    try:
        raw = generate_plan(schema_info, terms, question)
    except Exception as e:
        # 计划这一步失败不该让问答失败：记一条 warn，按普通方式继续
        update["plan_issues"] = [{
            "level": "warn", "kind": "error",
            "message": f"计划生成失败（{type(e).__name__}），按普通方式生成"}]
        return update

    plan = parse_plan(raw)
    if not plan:
        update["plan_issues"] = [{
            "level": "warn", "kind": "parse",
            "message": "计划没有返回可解析的 JSON，按普通方式生成"}]
        return update

    issues = validate_plan(plan, ctx.db_columns, ctx.enums, ctx.date_coverage, question,
                           skip_anchor_check=bool(intent))
    update.update({"plan": plan, "plan_used": True, "plan_issues": issues})

    for issue in issues:
        if issue.get("level") != "fatal":
            continue
        kind = issue.get("kind")
        if kind == "time":
            update["time_defer"] = issue["message"]
        elif kind == "enum":
            update["unknown_enum"] = issue.get("payload")
        elif kind == "anchor":
            update["defer"] = issue["message"]

    if not has_fatal(issues):
        update["plan_hint"] = format_plan_hint(plan, issues)
    return update


def make_plan_node(ctx: AgentContext):
    """状态机节点：理解之后、生成之前。"""

    def node(state: AgentState) -> dict:
        question = state.get("rewritten_question") or state.get("question", "")
        return build_plan(
            ctx, question,
            route=state.get("route", "sql"),
            schema_info=state.get("schema_info", ""),
            terms=state.get("terms", ""),
            has_history=bool(state.get("history")),
            intent=state.get("attribution_intent"),
            anchors=ctx.time_anchors(question),
        )

    return node
