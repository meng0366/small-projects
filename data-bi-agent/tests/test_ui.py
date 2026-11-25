"""Streamlit 界面的离线测试（不调用 LLM，不需要 API Key）。

用 Streamlit 官方的 AppTest 在进程内跑真实脚本：
- 首屏能正常渲染（标题、输入框、示例问题按钮）；
- 提交问题后能把状态渲染成「洞察 + 表格 + 图表 + 下载按钮」；
- 分析失败时给出警告而不是崩溃。

LLM 调用被 mock 掉，所以整组用例零成本、可进 CI。
"""
import os
import unittest
from unittest import mock

import pandas as pd

try:
    import _env  # noqa: F401  测试环境隔离（临时 AGENT_HOME、关闭 SQL 复用）
except ModuleNotFoundError:      # 以包方式运行（python -m unittest tests.test_xxx）
    from . import _env  # noqa: F401
except ModuleNotFoundError:      # 以包方式运行（python -m unittest tests.test_xxx）
    from . import _env  # noqa: F401

import app.config as app_config

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # 未安装 streamlit 时跳过，不影响其他测试
    AppTest = None

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "main.py"
)

FAKE_DF = pd.DataFrame({
    "区域": ["华南", "西南", "华东"],
    "销售额": [5064245.40, 4986839.00, 993052.95],
})

FAKE_STATE = {
    "question": "各区域的销售额是多少？",
    "rewritten_question": "各区域的销售额是多少？",
    "route": "sql",
    "statement": "SELECT c.region, SUM(o.amount) FROM orders o JOIN customers c "
                 "ON o.customer_id = c.customer_id GROUP BY c.region",
    "result": FAKE_DF,
    "insight": "华南以 506.4 万元居首，华东最低。",
    "answer": "华南以 506.4 万元居首，华东最低。",
    "errors": [],
    "retry_count": 0,
}

FAILED_STATE = {
    "question": "各区域的销售额是多少？",
    "route": "pandas",
    "statement": "result = df['nope'].sum()",
    "result": None,
    "insight": "",
    "answer": "未能完成分析。最后一次错误：KeyError: 'nope'（共尝试 3 次）",
    "errors": ["代码执行失败（runtime）：KeyError: 'nope'"],
    "retry_count": 2,
}


@unittest.skipIf(AppTest is None, "streamlit 未安装")
class TestWebUI(unittest.TestCase):
    """界面渲染与交互的离线用例。"""

    def setUp(self):
        # 界面在没配 Key 时会在 main() 里 st.stop()，整组用例都会挂。
        # 本机有 .env 时看不出来，CI（干净环境，没有 .env）必然全红。
        # 这里把 Key 换成占位值，让用例只依赖"代码"而不依赖"谁的机器"；
        # GraphAgent.run 全程被 mock，占位 Key 不会被真正拿去发请求。
        patcher = mock.patch.object(app_config, "LLM_API_KEY", "sk-offline-placeholder")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _app(self) -> "AppTest":
        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        return at

    def test_first_screen_renders(self):
        at = self._app()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("数据分析 Agent", at.title[0].value)
        self.assertEqual(len(at.chat_input), 1)
        # 侧边栏：7 个示例问题 + 清空对话
        self.assertEqual(len(at.sidebar.button), 8)

    def test_attribution_renders_detail_table(self):
        """归因场景展示的是"分项变化"表，而不是原始交叉表。"""
        at = self._app()
        state = dict(FAKE_STATE, attribution={
            "ok": True, "dim": "品类", "metric": "销售额",
            "base_label": "5月", "current_label": "6月",
            "additive": True,
            "table": pd.DataFrame({"维度": ["笔记本", "配件"],
                                   "基期": [674130.75, 31522.00],
                                   "本期": [889779.30, 47014.15],
                                   "变化": [215648.55, 15492.15],
                                   "贡献%": [48.4, 3.5]}),
        })
        with mock.patch("app.graph.GraphAgent.run", return_value=state):
            at.chat_input[0].set_value("为什么 6 月的销售额比 5 月高？").run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.dataframe), 1)
        captions = " ".join(c.value for c in at.caption)
        self.assertIn("归因明细", captions)

    def test_ask_renders_insight_table_chart(self):
        at = self._app()
        with mock.patch("app.graph.GraphAgent.run", return_value=FAKE_STATE):
            at.chat_input[0].set_value("各区域的销售额是多少？").run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.chat_message), 2)          # 一问一答
        self.assertEqual(len(at.dataframe), 1)
        self.assertEqual(len(at.get("download_button")), 1)
        self.assertEqual(len(at.get("plotly_chart")), 1)   # 类别+数值 → 柱状图
        self.assertTrue(any("华南" in m.value for m in at.markdown))
        self.assertTrue(any("Text-to-SQL" in c.value for c in at.caption))

    def test_failed_answer_shows_warning(self):
        at = self._app()
        with mock.patch("app.graph.GraphAgent.run", return_value=FAILED_STATE):
            at.chat_input[0].set_value("各区域的销售额是多少？").run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.dataframe), 0)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("未能完成分析", at.warning[0].value)

    def test_deferred_answer_is_not_a_warning(self):
        """拒答是"正常结论"：用正文展示，不要套失败样式吓人。"""
        at = self._app()
        deferred = {
            "question": "每个商品的毛利率是多少？",
            "route": "sql",
            "result": None,
            "defer": "缺少成本/进价字段",
            "answer": "这个问题用现有数据答不了，也不该编一个数字给你。\n\n原因：缺少成本/进价字段",
            "errors": [],
            "retry_count": 0,
        }
        with mock.patch("app.graph.GraphAgent.run", return_value=deferred):
            at.chat_input[0].set_value("每个商品的毛利率是多少？").run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.dataframe), 0)
        self.assertEqual(len(at.warning), 0, "拒答不该走失败告警样式")
        self.assertTrue(any("答不了" in m.value for m in at.markdown))
        self.assertTrue(any("没有给数字" in c.value for c in at.caption))

    def test_multi_turn_shows_rewrite(self):
        at = self._app()
        follow_up = dict(FAKE_STATE, question="只看华东",
                         rewritten_question="华东区域的销售额是多少？")
        seen = []

        def fake_run(question, history):
            # 在调用当刻记录，避免事后读同一个列表对象被后续会话改写
            seen.append((question, len(history)))
            return FAKE_STATE if len(seen) == 1 else follow_up

        with mock.patch("app.graph.GraphAgent.run", side_effect=fake_run):
            at.chat_input[0].set_value("各区域的销售额是多少？").run()
            at.chat_input[0].set_value("只看华东").run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.chat_message), 4)          # 两问两答
        # 第二轮必须把上一轮的一问一答传进状态机，否则追问无法改写
        self.assertEqual(seen, [("各区域的销售额是多少？", 0), ("只看华东", 2)])
        self.assertTrue(any("结合上下文理解为" in c.value for c in at.caption))

    def test_suggestion_buttons_start_next_question(self):
        """追问建议按钮点一下就能提问（pending → 重跑 → 正常问答链路）。"""
        at = self._app()
        state_with_suggestions = dict(
            FAKE_STATE,
            suggestions=["各品类的销售额是多少？", "每个月的销售额走势如何？"],
        )
        asked = []

        def fake_run(question, history):
            asked.append(question)
            return state_with_suggestions

        with mock.patch("app.graph.GraphAgent.run", side_effect=fake_run):
            at.chat_input[0].set_value("各区域的销售额是多少？").run()
            self.assertEqual(len(at.exception), 0)
            self.assertTrue(any("可以接着问" in c.value for c in at.caption))
            # 消息顺序是「用户(idx 0) + 助手(idx 1)」，建议按钮 key 用助手消息下标
            at.button(key="sug_1_0").click().run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(asked, ["各区域的销售额是多少？", "各品类的销售额是多少？"])

    def test_trace_id_shown_for_troubleshooting(self):
        """追踪号要出现在页面上，用户才能拿它去日志里查各阶段耗时（P2-2）。"""
        at = self._app()
        state = dict(FAKE_STATE, trace_id="20261124-101010-abcdefef")
        with mock.patch("app.graph.GraphAgent.run", return_value=state):
            at.chat_input[0].set_value("各区域的销售额是多少？").run()

        self.assertEqual(len(at.exception), 0)
        captions = " ".join(c.value for c in at.caption)
        self.assertIn("20261124-101010-abcdefef", captions)
        self.assertIn("logs --trace", captions)


if __name__ == "__main__":
    unittest.main()
