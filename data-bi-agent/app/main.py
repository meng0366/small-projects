"""Streamlit Web 界面（对应功能 F4.1–F4.4，阶段 4 实现）。

启动方式（在项目根目录执行）：

    streamlit run app/main.py

设计要点：
- **复用状态机**：直接用 app/graph.py 的 GraphAgent，界面只负责收问题、渲染状态，
  不重复业务逻辑，命令行与网页行为一致；
- **图表在界面侧现算**：以 with_chart=False 调用 Agent（不往 output/ 写 HTML），
  渲染时按结果集重新选图 → 页面内嵌交互图，也不留垃圾文件；
- **会话状态自管**：messages 存展示用消息（含 DataFrame），history 存给模型的多轮上下文，
  两者分开，避免把整张结果表塞进 Prompt。
"""
import os
import sys

# Streamlit 会把脚本所在目录（app/）放在 sys.path 首位，导致 `import app.xxx` 失败。
# 这里把项目根目录补进 sys.path，保证 `streamlit run app/main.py` 在任何工作目录下都能跑。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

from app.config import LLM_MODEL, MAX_QUESTIONS_PER_SESSION, check_config  # noqa: E402
from app.context import AgentContext  # noqa: E402
from app.graph import GraphAgent  # noqa: E402
from app.schema.loader import list_tables  # noqa: E402
from app.tools.chart_tool import CHART_TYPE_LABEL, build_chart  # noqa: E402
from app.tools.sql_tool import run_sql  # noqa: E402

EXAMPLE_QUESTIONS = [
    "各区域的销售额是多少？",
    "每个月的销售额走势如何？",
    "销售额最高的前 5 个商品是什么？",
    "哪个品类的客单价最高？",
    "哪个客户等级的销售额最高？",
    "华东区域每个月的销售额走势如何？",
]

ROUTE_LABEL = {
    "sql": "SQLite 多表 · Text-to-SQL",
    "pandas": "单表宽表 · Text-to-Pandas",
}


# ---------------------------------------------------------------- 资源加载
@st.cache_resource(show_spinner=False)
def get_ctx() -> AgentContext:
    """数据上下文只在进程内加载一次（读库 + 读语义词典）。"""
    return AgentContext()


@st.cache_resource(show_spinner="正在初始化模型与状态机…")
def get_agent(with_insight: bool) -> GraphAgent:
    """按"是否生成洞察"缓存状态机；共用同一个 AgentContext。"""
    return GraphAgent(get_ctx(), with_chart=False, with_insight=with_insight)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("asked", 0)


# ---------------------------------------------------------------- 侧边栏
def render_sidebar(ctx: AgentContext) -> tuple:
    with st.sidebar:
        st.subheader("数据源")
        st.markdown(f"**{ctx.mode_label}**")

        if ctx.has_db:
            for name in list_tables(ctx.db_path):
                r = run_sql(f'SELECT COUNT(*) AS c FROM "{name}"', ctx.db_path)
                cnt = int(r["df"].iloc[0, 0]) if r["ok"] else "?"
                st.caption(f"· `{name}` — {cnt} 行")
            schema_text = ctx.db_schema
        else:
            st.caption(f"· 宽表 `sales_orders.csv` — {ctx.df.shape[0]} 行 × {ctx.df.shape[1]} 列")
            schema_text = ctx.df_schema

        with st.expander("查看数据结构（随问题发给模型）"):
            st.code(schema_text, language="text")

        st.divider()
        st.subheader("选项")
        show_chart = st.toggle("自动生成图表", value=True,
                               help="按结果集结构自动选折线 / 柱状 / 饼图")
        with_insight = st.toggle("生成洞察结论", value=True,
                                 help="关闭可省一次模型调用")
        show_code = st.toggle("显示生成的 SQL / 代码", value=True)

        st.divider()
        st.subheader("示例问题")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if st.button(q, key=f"ex_{i}", width="stretch"):
                st.session_state.pending = q

        st.divider()
        if st.button("清空对话", width="stretch"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

        if MAX_QUESTIONS_PER_SESSION:
            used = st.session_state.get("asked", 0)
            st.caption(f"本次会话已提问 {used} / {MAX_QUESTIONS_PER_SESSION} 次")
        st.caption(f"模型：{LLM_MODEL}　|　后端：LangGraph 状态机")

    return show_chart, with_insight, show_code


# ---------------------------------------------------------------- 问答
def answer_question(agent: GraphAgent, question: str) -> None:
    """执行一次问答，把结果存入会话状态。"""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.asked = st.session_state.get("asked", 0) + 1

    with st.spinner("正在分析…"):
        try:
            state = agent.run(question, st.session_state.history)
        except Exception as e:  # 网络 / API 异常时不让页面崩掉
            state = {"answer": f"调用失败：{type(e).__name__}: {e}", "errors": [str(e)]}

    st.session_state.messages.append({
        "role": "assistant",
        "question": question,
        "rewritten": state.get("rewritten_question") or "",
        "route": state.get("route") or "",
        "statement": state.get("statement") or "",
        "df": state.get("result"),
        "insight": state.get("insight") or "",
        "answer": state.get("answer") or "",
        "errors": state.get("errors") or [],
        "retry": state.get("retry_count", 0),
        "deferred": bool(state.get("defer")),
        "suggestions": state.get("suggestions") or [],
    })

    # 多轮上下文只保留文字，不带表格（控制 token）
    # 注意用拼接生成新列表而不是 +=：+= 会原地修改，把已经传给 agent 的列表也一起改了
    st.session_state.history = (st.session_state.history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": state.get("insight") or state.get("answer") or ""},
    ])[-8:]   # 最近 4 轮


# ---------------------------------------------------------------- 渲染
def render_assistant(msg: dict, show_chart: bool, show_code: bool, idx: int = 0) -> None:
    df = msg.get("df")
    question = msg.get("question", "")

    if df is None:
        # 拒答是"正常结论"（这题确实答不了），按正文展示；真正的失败才用警示样式
        if msg.get("deferred"):
            st.markdown(msg.get("answer") or "这个问题用现有数据答不了。")
            st.caption("没有给数字，是因为给出数字就意味着换了个口径或换了个范围")
        else:
            st.warning(msg.get("answer") or "未能完成分析。")
        for err in msg.get("errors") or []:
            st.caption(f"· {err}")
        return

    if msg.get("rewritten") and msg["rewritten"] != question:
        st.caption(f"结合上下文理解为：{msg['rewritten']}")

    if msg.get("insight"):
        st.markdown(msg["insight"])

    meta = [ROUTE_LABEL.get(msg.get("route", ""), ""), f"{len(df)} 行 × {df.shape[1]} 列"]
    if msg.get("retry"):
        meta.append(f"自动重试 {msg['retry']} 次后完成")
    st.caption("　|　".join(m for m in meta if m))

    if len(df) == 0:
        # 空结果不画空图、不给空表，直接说明（洞察文案里已写清原因与建议）
        st.info("当前条件下没有匹配的数据行。")
    else:
        if show_chart:
            fig, chart_type = build_chart(df, question)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
                st.caption(f"图表类型：{CHART_TYPE_LABEL.get(chart_type, chart_type)}")

        st.dataframe(df, width="stretch",
                     height=min(420, 42 + 35 * min(len(df), 10)))
        st.download_button(
            "下载结果 CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="result.csv",
            mime="text/csv",
            key=f"dl_{msg.get('question', '')}_{len(df)}",
        )

    if show_code and msg.get("statement"):
        label = "SQL" if msg.get("route") == "sql" else "Python"
        with st.expander(f"查看生成的 {label}"):
            st.code(msg["statement"], language="sql" if label == "SQL" else "python")

    if msg.get("retry"):
        with st.expander(f"自动重试记录（{msg['retry']} 次）"):
            for err in msg.get("errors") or []:
                st.caption(f"· {err}")

    # 追问建议：点一下直接接着问，省去自己组织语言（规则生成，不额外耗 token）
    suggestions = msg.get("suggestions") or []
    if suggestions:
        st.caption("可以接着问：")
        columns = st.columns(len(suggestions))
        for j, text in enumerate(suggestions):
            with columns[j]:
                if st.button(text, key=f"sug_{idx}_{j}", width="stretch"):
                    st.session_state.pending = text
                    st.rerun()


# ---------------------------------------------------------------- 主流程
def main() -> None:
    st.set_page_config(page_title="数据分析 Agent", page_icon="📊", layout="wide")
    init_state()

    st.title("📊 数据分析 Agent")
    st.caption("用自然语言提问，自动生成 SQL 或 pandas 代码、执行、出图并给出结论。")

    if not check_config():
        st.error("未配置 API Key。请把 `.env.example` 复制为 `.env`，"
                 "填入 `DEEPSEEK_API_KEY` 后刷新页面。")
        st.stop()

    ctx = get_ctx()
    show_chart, with_insight, show_code = render_sidebar(ctx)
    agent = get_agent(with_insight)

    # st.chat_input 固定吸在页面底部，与调用位置无关，所以先收问题再渲染历史消息，
    # 新问答当次就能出现在页面上，无需 st.rerun()（多一次重跑会让问答被执行两次）。
    pending = st.session_state.pending
    st.session_state.pending = None
    typed = st.chat_input("输入你的问题，例如：各区域的销售额是多少？")
    if pending or typed:
        if (MAX_QUESTIONS_PER_SESSION
                and st.session_state.get("asked", 0) >= MAX_QUESTIONS_PER_SESSION):
            st.warning(f"本次会话已达到提问上限（{MAX_QUESTIONS_PER_SESSION} 次）。"
                       "刷新页面可重置，或调整环境变量 `AGENT_MAX_QUESTIONS`。")
        else:
            answer_question(agent, pending or typed)

    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🧑" if msg["role"] == "user" else "📊"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg, show_chart, show_code, idx)

    if not st.session_state.messages:
        st.info("在下方输入问题，或点左侧示例问题快速体验。")


if __name__ == "__main__":
    main()
