"""取值校验（enum_guard）与澄清分支的测试（P0-1b）。

真实试跑里模型把「办公用品」映射成库里的「配件」、「电脑」映射成「笔记本」，
都给出了看起来正常的金额 —— 这类错误不报错、不返空，用户完全看不出来。
这里锁住两条底线：
1. 用户提到的值库里没有 → 必须澄清，不许生成；
2. 正常问题（值合法、泛指维度）→ 绝不能误拦。
"""
import unittest
from unittest import mock

import app.graph as G
from app.context import AgentContext
from app.tools.enum_guard import build_clarify_answer, find_unknown_enum

GOOD_SQL = ("SELECT p.category AS 品类, ROUND(SUM(o.amount), 2) AS 销售额 "
            "FROM orders o JOIN products p ON o.product_id = p.product_id "
            "GROUP BY p.category")


def _enums():
    return AgentContext().data_profile["enums"]


class TestFindUnknownEnum(unittest.TestCase):
    """规则本身：该拦的拦住，不该拦的放行。"""

    def test_unknown_value_is_caught(self):
        cases = {
            "办公用品类产品的销售额是多少？": "办公用品",
            "电脑类产品一共卖了多少？": "电脑",
            "东北地区一共有多少个客户？": "东北",
            "帮我看看家用电器类的销售额": "家用电器",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                flag = find_unknown_enum(question, _enums())
                self.assertIsNotNone(flag, f"应识别出未知取值：{question}")
                self.assertEqual(flag["candidate"], expected)

    def test_valid_or_generic_questions_pass(self):
        """合法取值、泛指维度、固定搭配都不能误报。"""
        questions = [
            "华东地区的销售额是多少？",     # 合法取值
            "手机类产品卖了多少？",          # 合法取值
            "各品类的销售额是多少？",        # 泛指（"各"）
            "按照品类看销售额",              # 泛指（"按照"）
            "销售额最高的品类是什么？",      # 泛指（"最高的"）
            "各客户等级的销售额",            # 固定搭配"客户"+"等级"
            "客户等级分布",
            "这类产品卖得怎么样",
            "西南区的客户有多少",            # 线索词未命中，宁可漏报
            "2025 年总销售额是多少？",
            "销售额最高的前 5 个客户",
            "平板和手机哪个卖得好",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(find_unknown_enum(question, _enums()))

    def test_empty_input_safe(self):
        self.assertIsNone(find_unknown_enum("", _enums()))
        self.assertIsNone(find_unknown_enum("办公用品类", []))
        self.assertIsNone(find_unknown_enum("办公用品类", [{"values": []}]))

    def test_clarify_answer_lists_real_values(self):
        flag = find_unknown_enum("办公用品类产品的销售额是多少？", _enums())
        answer = build_clarify_answer(flag)
        self.assertIn("办公用品", answer)
        self.assertIn("配件", answer)      # 必须给出真实可选值
        self.assertIn("口径", answer)      # 说明为什么不硬映射


class TestClarifyFlow(unittest.TestCase):
    """端到端：澄清分支拦在生成之前，且正常问题不受影响。"""

    def test_unknown_category_never_reaches_generate(self):
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        generate = mock.MagicMock()
        with mock.patch.object(G, "generate_sql", generate):
            state = agent.run("办公用品类产品的销售额是多少？")

        generate.assert_not_called()                  # 一次模型都没调
        self.assertIsNone(state["result"])
        self.assertIn("办公用品", state["answer"])
        self.assertIn("配件", state["answer"])
        self.assertEqual(state["chart_type"], "none")
        # 可选值直接做成能点的问题
        self.assertTrue(state["suggestions"])
        self.assertTrue(all(s.endswith("？") for s in state["suggestions"]))

    def test_unknown_region_also_clarified(self):
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        generate = mock.MagicMock()
        with mock.patch.object(G, "generate_sql", generate):
            state = agent.run("东北地区一共有多少个客户？")
        generate.assert_not_called()
        self.assertIn("东北", state["answer"])

    def test_valid_question_goes_to_generate(self):
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_sql", return_value=GOOD_SQL):
            state = agent.run("手机类产品的销售额是多少？")
        self.assertIsNone(state.get("unknown_enum"))
        self.assertIsNotNone(state["result"])


if __name__ == "__main__":
    unittest.main()
