"""指标归因（P1-2）的离线测试：不调 LLM，全部走纯函数与 mock。

覆盖三类：
1. 意图识别：谁是基期、谁是被解释对象，以及"什么时候不该当归因"；
2. 结果拆解：差分、占比、排序、闭合校验、多年数据的年份歧义、比率指标；
3. 状态机接入：能拆就出归因结论，拆不开如实降级，普通问题不被误判成归因。
"""
import unittest
from unittest import mock

import pandas as pd

import app.graph as G
from app.tools.attribution import (
    attribution_chart_frame,
    decompose,
    detect_attribution,
    format_attribution,
    period_key,
)

ENUMS = [{"table": "customers", "column": "region",
          "values": ["华东", "华南", "华北", "西南", "华中"]}]


def crosstab() -> pd.DataFrame:
    """一张「月份 × 品类」交叉表：6 月总额比 5 月多 8。"""
    return pd.DataFrame({
        "月份": ["2025-05"] * 5 + ["2025-06"] * 5,
        "品类": ["手机", "笔记本", "平板", "配件", "智能家居"] * 2,
        "销售额": [10.0, 20, 30, 40, 50, 15, 18, 45, 20, 60],
    })


def crosstab_two_dims() -> pd.DataFrame:
    """「月份 × 品类 × 客户等级」四列交叉表：6 月比 5 月少 20，笔记本拖了 25。"""
    rows = [
        ("2025-05", "手机", "普通客户", 10.0), ("2025-05", "手机", "会员客户", 20.0),
        ("2025-05", "笔记本", "普通客户", 40.0), ("2025-05", "笔记本", "会员客户", 20.0),
        ("2025-05", "平板", "普通客户", 10.0), ("2025-05", "平板", "会员客户", 5.0),
        ("2025-06", "手机", "普通客户", 10.0), ("2025-06", "手机", "会员客户", 20.0),
        ("2025-06", "笔记本", "普通客户", 20.0), ("2025-06", "笔记本", "会员客户", 15.0),
        ("2025-06", "平板", "普通客户", 12.0), ("2025-06", "平板", "会员客户", 8.0),
    ]
    return pd.DataFrame(rows, columns=["月份", "品类", "客户等级", "销售额"])


class TestPeriodKey(unittest.TestCase):
    def test_variants(self):
        cases = {
            "2025-06": (2025, 6), "2025/6": (2025, 6), "2025年6月": (2025, 6),
            "六月": (None, 6), "6月": (None, 6), "202506": (2025, 6),
            "2025年": (2025, None), "手机": None, "": None,
        }
        for text, expected in cases.items():
            self.assertEqual(period_key(text), expected, text)


class TestIntentDetection(unittest.TestCase):
    def test_than_order_puts_explained_subject_current(self):
        """「6 月比 5 月高」→ 解释 6 月，参照 5 月。"""
        intent = detect_attribution("为什么 2025 年 6 月的销售额比 5 月高？", ENUMS)
        self.assertEqual(intent["kind"], "period")
        self.assertEqual(intent["current"], "2025年6月")
        self.assertEqual(intent["base"], "5月")
        self.assertFalse(intent["assumed"])

    def test_from_to_order(self):
        intent = detect_attribution("从 5 月到 6 月销售额为什么上升了？", ENUMS)
        self.assertEqual((intent["base"], intent["current"]), ("5月", "6月"))

    def test_single_period_defaults_to_previous_month(self):
        """只说了一个月：默认与上一期比，但必须标记为假设，好在结论里写明。"""
        intent = detect_attribution("6 月的销售额为什么下降了？", ENUMS)
        self.assertTrue(intent["assumed"])
        self.assertEqual(intent["base"], "5月")
        self.assertEqual(intent["base_key"], (None, 5))

    def test_january_wraps_to_previous_december(self):
        intent = detect_attribution("为什么 2025 年 1 月销售额下降了？", ENUMS)
        self.assertEqual(intent["base_key"], (2024, 12))

    def test_dimension_pair(self):
        intent = detect_attribution("为什么华东的销售额比华北高？", ENUMS)
        self.assertEqual(intent["kind"], "dimension")
        self.assertEqual((intent["base"], intent["current"]), ("华北", "华东"))
        self.assertEqual(intent["column"], "region")

    def test_plain_question_is_not_attribution(self):
        for question in ("各区域的销售额是多少？", "为什么销售额最高的品类是手机？",
                         "2025 年 6 月的销售额是多少？"):
            self.assertIsNone(detect_attribution(question, ENUMS), question)

    def test_cause_word_without_comparison_is_not_attribution(self):
        self.assertIsNone(detect_attribution("为什么销售额这么高？", ENUMS))


class TestDecompose(unittest.TestCase):
    def setUp(self):
        self.intent = detect_attribution("为什么 2025 年 6 月的销售额比 5 月高？", ENUMS)

    def test_contributions_add_up(self):
        att = decompose(crosstab(), self.intent)
        self.assertTrue(att["ok"])
        self.assertEqual(att["base_total"], 150.0)
        self.assertEqual(att["current_total"], 158.0)
        self.assertEqual(att["delta"], 8.0)
        self.assertAlmostEqual(att["table"]["变化"].sum(), 8.0, places=2)
        self.assertAlmostEqual(att["table"]["贡献%"].sum(), 100.0, places=1)

    def test_sorted_by_absolute_change(self):
        att = decompose(crosstab(), self.intent)
        changes = att["table"]["变化"].abs().tolist()
        self.assertEqual(changes, sorted(changes, reverse=True))
        self.assertEqual(att["table"].iloc[0]["维度"], "配件")   # -20 变化最大

    def test_total_row_is_excluded(self):
        """结果里混进合计行时，分项会被翻倍 —— 必须剔除。"""
        frame = pd.concat([crosstab(), pd.DataFrame(
            {"月份": ["2025-06"], "品类": ["合计"], "销售额": [158.0]})], ignore_index=True)
        att = decompose(frame, self.intent)
        self.assertTrue(att["ok"])
        self.assertNotIn("合计", att["table"]["维度"].tolist())
        self.assertEqual(att["current_total"], 158.0)

    def test_missing_side_filled_with_zero_and_disclosed(self):
        frame = crosstab()
        frame = frame[~((frame["月份"] == "2025-06") & (frame["品类"] == "配件"))]
        att = decompose(frame, self.intent)
        self.assertTrue(att["ok"])
        row = att["table"][att["table"]["维度"] == "配件"].iloc[0]
        self.assertEqual(row["本期"], 0.0)
        self.assertTrue(any("配件" in n and "只在" in n for n in att["notes"]))

    def test_ratio_metric_is_not_additive(self):
        frame = crosstab().rename(columns={"销售额": "客单价"})
        att = decompose(frame, self.intent, "6 月客单价比 5 月高吗")
        self.assertFalse(att["additive"])
        self.assertNotIn("贡献%", att["table"].columns)
        self.assertIn("不能相加", format_attribution(att))

    def test_only_totals_cannot_be_decomposed(self):
        frame = pd.DataFrame({"月份": ["2025-05", "2025-06"], "销售额": [150.0, 158.0]})
        att = decompose(frame, self.intent)
        self.assertFalse(att["ok"])
        self.assertIn("维度", att["reason"])

    def test_unknown_compare_label_fails_honestly(self):
        intent = detect_attribution("为什么 2025 年 3 月的销售额比 2 月高？", ENUMS)
        att = decompose(crosstab(), intent)
        self.assertFalse(att["ok"])
        self.assertTrue(att["reason"])

    def test_multi_year_data_uses_latest_year(self):
        """「6 月」在跨年数据里会同时命中两年，必须只取最新一年并说明。"""
        frame = pd.DataFrame({
            "月份": ["2024-05", "2024-06", "2025-05", "2025-06"] * 2,
            "品类": ["手机"] * 4 + ["平板"] * 4,
            "销售额": [1.0, 2, 100, 150, 5.0, 6, 10, 20],
        })
        intent = detect_attribution("为什么 6 月销售额比 5 月高？", ENUMS)
        att = decompose(frame, intent)
        self.assertTrue(att["ok"])
        self.assertEqual(att["base_total"], 110.0)      # 100 + 10，只算 2025
        self.assertEqual(att["current_total"], 170.0)
        self.assertTrue(any("最新" in n for n in att["notes"]))

    def test_february_days_are_flagged(self):
        frame = pd.DataFrame({
            "月份": ["2025-02", "2025-03"] * 2,
            "品类": ["手机", "手机", "平板", "平板"],
            "销售额": [10.0, 30, 10, 30],
        })
        intent = detect_attribution("为什么 3 月销售额比 2 月高？", ENUMS)
        att = decompose(frame, intent)
        self.assertTrue(any("天数" in n for n in att["notes"]))


class TestTwoDimDrill(unittest.TestCase):
    """二级拆解：用户点两个维度时，对变化最大的一项再拆一层。"""

    def _intent(self, q="为什么 6 月销售额比 5 月低？按品类和客户等级拆"):
        return detect_attribution(q, ENUMS)

    def test_hints_follow_text_order(self):
        """先说哪个哪个是一级 —— 不能按字典顺序碰运气。"""
        intent = detect_attribution("为什么 6 月销售额比 5 月低？按客户等级和品类拆", ENUMS)
        self.assertEqual(intent["dimension_hints"], ["客户等级", "品类"])
        self.assertEqual(intent["dimension_hint"], "客户等级")

    def test_single_hint_stays_single(self):
        intent = detect_attribution("为什么 6 月销售额比 5 月低？按品类拆", ENUMS)
        self.assertEqual(intent["dimension_hints"], ["品类"])

    def test_drill_happens_on_largest_item(self):
        att = decompose(crosstab_two_dims(), self._intent())
        self.assertTrue(att["ok"])
        self.assertEqual(att["dim"], "品类")
        self.assertEqual(att["dim2"], "客户等级")
        drill = att["drill"]
        self.assertIsNotNone(drill)
        self.assertEqual(drill["item"], "笔记本")     # 变化 -25，绝对值最大
        self.assertEqual(set(drill["table"]["维度"]), {"普通客户", "会员客户"})

    def test_drill_closes_back_to_item(self):
        """二级分项加总必须回到该项的变化，否则下钻时漏了行。"""
        att = decompose(crosstab_two_dims(), self._intent())
        drill = att["drill"]
        self.assertAlmostEqual(drill["table"]["变化"].sum(), drill["item_delta"], places=2)
        self.assertTrue(drill["closure_ok"])

    def test_drill_share_uses_total_direction(self):
        """总额在跌：跟着跌的二级分项显示为正贡献（占总降幅的百分比）。"""
        att = decompose(crosstab_two_dims(), self._intent())
        row = att["drill"]["table"].iloc[0]
        self.assertEqual(row["维度"], "普通客户")
        self.assertAlmostEqual(row["贡献%"], 100.0, places=1)   # -20 / 总降幅 20

    def test_no_second_dim_means_no_drill(self):
        att = decompose(crosstab(), detect_attribution("为什么 6 月销售额比 5 月高？按品类拆", ENUMS))
        self.assertTrue(att["ok"])
        self.assertIsNone(att["drill"])
        self.assertEqual(att["dim2"], "")

    def test_requested_two_layers_but_missing_explains(self):
        """点了两层却只有一层数据：要说明，别让用户以为已经拆到最细。"""
        att = decompose(crosstab(), self._intent("为什么 6 月销售额比 5 月高？按品类和客户等级拆"))
        self.assertTrue(att["ok"])
        self.assertIsNone(att["drill"])
        self.assertTrue(any("两层" in n for n in att["notes"]))

    def test_summary_has_drill_block(self):
        text = format_attribution(decompose(crosstab_two_dims(), self._intent()))
        self.assertIn("二级拆解", text)
        self.assertIn("笔记本", text)
        self.assertIn("加总", text)


class TestFormat(unittest.TestCase):
    def _att(self):
        """总额下降 10：手机 -20、平板 +10（有涨有跌，两个方向都要能讲）。"""
        frame = pd.DataFrame({
            "月份": ["2025-05"] * 2 + ["2025-06"] * 2,
            "品类": ["手机", "平板"] * 2,
            "销售额": [100.0, 0.0, 80.0, 10.0],
        })
        intent = detect_attribution("为什么 6 月销售额比 5 月低？", ENUMS)
        return decompose(frame, intent)

    def test_decline_leads_with_drag_items(self):
        """总额在跌时先讲拖累项，否则读者会以为结论是"在涨"。"""
        text = format_attribution(self._att())
        self.assertIn("减少", text)
        self.assertIn("主要拖累", text)
        self.assertIn("反向拉动", text)
        self.assertLess(text.index("主要拖累"), text.index("反向拉动"))

    def test_closure_statement_present(self):
        text = format_attribution(self._att())
        self.assertIn("加总", text)
        self.assertIn("一致", text)

    def test_negative_total_keeps_same_direction_share_positive(self):
        """总额在跌时，跟着跌的分项应显示"占降幅 50%"，而不是 -50%。"""
        frame = pd.DataFrame({
            "月份": ["2025-05"] * 2 + ["2025-06"] * 2,
            "品类": ["手机", "平板"] * 2,
            "销售额": [100.0, 100.0, 50.0, 90.0],
        })
        intent = detect_attribution("为什么 6 月销售额比 5 月低？", ENUMS)
        att = decompose(frame, intent)
        row = att["table"][att["table"]["维度"] == "手机"].iloc[0]
        self.assertGreater(row["贡献%"], 0)
        self.assertIn("占降幅的", format_attribution(att))

    def test_offsetting_item_is_described_as_offset(self):
        frame = pd.DataFrame({
            "月份": ["2025-05"] * 2 + ["2025-06"] * 2,
            "品类": ["手机", "平板"] * 2,
            "销售额": [100.0, 100.0, 50.0, 120.0],
        })
        intent = detect_attribution("为什么 6 月销售额比 5 月低？", ENUMS)
        self.assertIn("对冲了", format_attribution(decompose(frame, intent)))

    def test_chart_frame_only_has_dimension_and_delta(self):
        frame = attribution_chart_frame(self._att())
        self.assertEqual(list(frame.columns), ["维度", "变化"])


SQL_OK = """SELECT strftime('%Y-%m', o.order_date) AS 月份, p.category AS 品类,
       ROUND(SUM(o.amount), 2) AS 销售额
FROM orders o JOIN products p ON o.product_id = p.product_id
WHERE strftime('%Y-%m', o.order_date) IN ('2025-05','2025-06')
GROUP BY 月份, 品类"""
SQL_TOTALS_ONLY = """SELECT strftime('%Y-%m', o.order_date) AS 月份,
       ROUND(SUM(o.amount), 2) AS 销售额
FROM orders o
WHERE strftime('%Y-%m', o.order_date) IN ('2025-05','2025-06')
GROUP BY 月份"""
SQL_FOUR_COLUMNS = """SELECT strftime('%Y-%m', o.order_date) AS 月份,
       p.category AS 品类, c.customer_level AS 客户等级,
       ROUND(SUM(o.amount), 2) AS 销售额
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE strftime('%Y-%m', o.order_date) IN ('2025-05','2025-06')
GROUP BY 月份, 品类, 客户等级"""


class TestGraphIntegration(unittest.TestCase):
    def _run(self, question, sql):
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_attribution_sql", return_value=sql) as att_mock, \
                mock.patch.object(G, "generate_sql", return_value=sql) as sql_mock:
            state = agent.run(question)
        return state, att_mock, sql_mock

    def test_attribution_question_uses_dedicated_prompt(self):
        state, att_mock, sql_mock = self._run("为什么 2025 年 6 月的销售额比 5 月高？", SQL_OK)
        self.assertTrue(att_mock.called)
        self.assertFalse(sql_mock.called)
        att = state["attribution"]
        self.assertTrue(att["ok"])
        self.assertEqual(att["dim"], "品类")
        self.assertIn("主要拉动", state["answer"])
        self.assertEqual(len(state["result"]), 10)

    def test_plain_question_never_enters_attribution(self):
        state, att_mock, sql_mock = self._run(
            "各区域的销售额是多少？",
            "SELECT c.region AS 区域, ROUND(SUM(o.amount),2) AS 销售额 FROM orders o "
            "JOIN customers c ON o.customer_id=c.customer_id GROUP BY c.region")
        self.assertFalse(att_mock.called)
        self.assertTrue(sql_mock.called)
        self.assertIsNone(state.get("attribution"))

    def test_undecomposable_result_degrades_with_reason(self):
        """模型只给了两期总额：不许硬凑归因，要说明原因并保留原始结果。"""
        state, _, _ = self._run("为什么 2025 年 6 月的销售额比 5 月高？", SQL_TOTALS_ONLY)
        self.assertFalse(state["attribution"]["ok"])
        self.assertIn("没能拆出归因", state["answer"])
        self.assertIsNotNone(state["result"])
        self.assertEqual(len(state["result"]), 2)

    def test_two_dimension_question_drills_into_answer(self):
        """点两个维度 → 拿四列表 → 一级拆解 + 二级下钻，结论里两段都要有。"""
        state, _, _ = self._run(
            "为什么 2025 年 6 月的销售额比 5 月低？按品类和客户等级拆", SQL_FOUR_COLUMNS)
        att = state["attribution"]
        self.assertTrue(att["ok"])
        self.assertIsNotNone(att["drill"])
        self.assertEqual(att["dim"], "品类")
        self.assertEqual(att["drill"]["dim"], "客户等级")
        self.assertIn("二级拆解", state["answer"])
        self.assertTrue(att["drill"]["closure_ok"])


if __name__ == "__main__":
    unittest.main()
