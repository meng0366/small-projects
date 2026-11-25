"""生成节点：pandas 代码（阶段 1）与 SQL 语句（阶段 2）。

对应功能：
- F2.1 Text-to-Pandas 代码生成
- F2.3 Text-to-SQL 生成
- F2.7 修正回流（把人工修正过的问答作为 few-shot 注入，P1-1）

借鉴点：
- PandasAI 的 code_generation 模块：用 schema + few-shot 生成代码，且不发送整表数据；
- OpenChatBI 的 schema 关联：把业务术语注入 Prompt，减少口径误解；
- Datus-agent 的 correction 回流：人工修正过的问答要沉淀下来，而不是只解决一次。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import get_llm, strip_code_fence
from app.prompts.templates import (
    EMPTY_RESULT_FEEDBACK_TEMPLATE,
    ERROR_FEEDBACK_TEMPLATE,
    FEW_SHOT_ATTRIBUTION,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_SQL,
    LESSON_PREAMBLE,
    SQL_USER_TEMPLATE,
    SYSTEM_ANALYST,
    SYSTEM_ATTRIBUTION,
    SYSTEM_SQL,
    USER_TEMPLATE,
)
from app.tools.attribution import describe_comparison


SCHEMA_BLOCK_HEADER = "数据结构与口径（唯一权威来源，下面的示例里不再重复）："
"""schema 块的引导语。放在 system 里只出现一次，见 `_system_with_context`。"""


def _system_with_context(system: str, schema_info: str, terms: str) -> str:
    """把 schema 与术语拼进 system 消息（整份 prompt 里只出现一次）。

    以前它们是 `user_template` 的占位符，于是**每条 few-shot 示例都带一份**：
    实测 schema 1299 字符 + 术语 198 字符，重复 4 次（3 条示例 + 当前问题），
    在 7313 字符的 prompt 里占了 5988 —— 82% 是纯重复。
    挪进 system 后：① 上下文长度大降；② system + 示例构成**跨问题不变的前缀**，
    DeepSeek 的上下文缓存在相同前缀上才命中，重复的部分本来也拿不到缓存收益；
    ③ 示例只剩"问题 → 答案"，更像在教"怎么写"，而不是在教"表长什么样"。

    刻意保留 `user_template` 里的 `{schema}` / `{terms}` 占位符兼容：
    `str.format` 对多余的关键字参数是忽略的，所以老的调用方式不会报错。
    """
    blocks = [system]
    if schema_info:
        blocks.append(f"{SCHEMA_BLOCK_HEADER}\n{schema_info}")
    if terms:
        blocks.append(terms)
    return "\n\n".join(blocks)


def _build_messages(system: str, examples: list, user_template: str,
                    schema_info: str, terms: str, question: str,
                    previous_error: str = None, kind: str = "SQL",
                    previous_empty: bool = False, lessons: list = None,
                    plan_hint: str = "") -> list:
    """构造带 few-shot 的消息列表；重试时追加反馈消息。

    previous_empty 与 previous_error 是两种不同的失败：
    - 报错：语法 / 字段名错了，按错误信息修正；
    - 空结果：执行成功但 0 行，通常是条件过严或枚举值写错，要放宽条件。
    用同一句"执行失败"提示会误导模型去修语法，所以分两套模板。

    plan_hint 是计划先行（P1）产出的引导语，只拼在**当前问题**后面，不进 few-shot
    —— 示例是"别的题怎么写"，把本地的计划混进去会让模型以为示例里也有这层约束。

    lessons 是修正回流（P1-1）：排在通用示例**之后**、当前问题**之前**。
    越靠近当前问题影响力越大，而它本来就是"这个问题该怎么写"的示范。
    第一条 lesson 前面还要加一句"已确认口径"——实测光给示例，模型仍会以
    "这个词没有定义"为由拒答，必须明说这条写法本身就是定义。
    """
    messages = [SystemMessage(content=_system_with_context(system, schema_info, terms))]

    pairs = list(examples or [])
    for q, output in pairs:
        messages.append(HumanMessage(
            content=user_template.format(schema=schema_info, terms=terms, question=q)))
        messages.append(AIMessage(content=output))

    for index, item in enumerate(lessons or []):
        prefix = LESSON_PREAMBLE if index == 0 else ""
        messages.append(HumanMessage(content=prefix + user_template.format(
            schema=schema_info, terms=terms, question=item.get("question", ""))))
        messages.append(AIMessage(content=item.get("output", "")))

    current = question + ("\n\n" + plan_hint if plan_hint else "")
    messages.append(HumanMessage(
        content=user_template.format(schema=schema_info, terms=terms, question=current)))

    if previous_empty:
        messages.append(HumanMessage(content=EMPTY_RESULT_FEEDBACK_TEMPLATE.format(kind=kind)))
    elif previous_error:
        messages.append(HumanMessage(content=ERROR_FEEDBACK_TEMPLATE.format(
            kind=kind, error=previous_error)))
    return messages


def generate_code(schema_info: str, terms: str, question: str,
                  previous_error: str = None, previous_empty: bool = False,
                  lessons: list = None, plan_hint: str = "") -> str:
    """根据问题 + schema + 术语生成 pandas 代码。"""
    messages = _build_messages(SYSTEM_ANALYST, FEW_SHOT_EXAMPLES, USER_TEMPLATE,
                               schema_info, terms, question,
                               previous_error, kind="Python 代码",
                               previous_empty=previous_empty, lessons=lessons,
                               plan_hint=plan_hint)
    resp = get_llm().invoke(messages)
    return strip_code_fence(resp.content)


def generate_sql(schema_info: str, terms: str, question: str,
                 previous_error: str = None, previous_empty: bool = False,
                 lessons: list = None, plan_hint: str = "") -> str:
    """根据问题 + 数据库 schema + 术语生成 SQL 语句（F2.3）。"""
    messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                               schema_info, terms, question,
                               previous_error, kind="SQL",
                               previous_empty=previous_empty, lessons=lessons,
                               plan_hint=plan_hint)
    resp = get_llm().invoke(messages)
    return strip_code_fence(resp.content)


def generate_attribution_sql(schema_info: str, terms: str, question: str, intent: dict,
                             previous_error: str = None, previous_empty: bool = False,
                             lessons: list = None, plan_hint: str = "") -> str:
    """归因专用：生成一条「对比列 × 拆解维度 × 指标」的交叉表 SQL（P1-2）。

    与普通 SQL 生成的区别只在提示词：这里必须明确要求三列交叉表，
    否则模型常常只返回"两期各一行"的总额，那种结果在代码侧拆不开
    （`decompose` 会判 ok=False 并降级），用户就得不到归因。
    """
    messages = _build_messages(SYSTEM_ATTRIBUTION, FEW_SHOT_ATTRIBUTION, SQL_USER_TEMPLATE,
                               schema_info, terms, question + describe_comparison(intent),
                               previous_error, kind="SQL",
                               previous_empty=previous_empty, lessons=lessons)
    resp = get_llm().invoke(messages)
    return strip_code_fence(resp.content)
