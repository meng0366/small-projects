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
            "2025-06": (2025, 6, 6), "2025/6": (2025, 6, 6), "2025年6月": (2025, 6, 6),
            "六月": (None, 6, 6), "6月": (None, 6, 6), "202506": (2025, 6, 6),
            "2025年": (2025, None, None), "手机": None, "": None,
        }
        for text, expected in cases.items():
            self.assertEqual(period_key(text), expected, text)

    def test_quarter_and_half_spans(self):
        """季度/半年是跨度，不是某个月：归一成 (年, 起, 止)。"""
        cases = {
            "Q2": (None, 4, 6), "q2": (None, 4, 6), "2025Q2": (2025, 4, 6),
            "2025-Q1": (2025, 1, 3), "第二季度": (None, 4, 6), "第 3 季度": (None, 7, 9),
            "2025年第四季度": (2025, 10, 12),
            "上半年": (None, 1, 6), "下半年": (None, 7, 12), "H1": (None, 1, 6),
            "2025年下半年": (2025, 7, 12), "2025年H2": (2025, 7, 12),
        }
        for text, expected in cases.items():
            self.assertEqual(period_key(text), expected, text)

    def test_quarter_month_pair_matches_by_overlap(self):
        """用户说 "Q1"、数据里是月份行：两者指同一段时间，要能对上。"""
        from app.tools.attribution import _compatible

        self.assertTrue(_compatible(period_key("Q1"), period_key("2025-02")))
        self.assertTrue(_compatible(period_key("Q1"), period_key("2025-03")))
        self.assertFalse(_compatible(period_key("Q1"), period_key("2025-04")))
        self.assertFalse(_compatible(period_key("2025Q2"), period_key("2024-05")))


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
        self.assertEqual(intent["base_key"], (None, 5, 5))

    def test_january_wraps_to_previous_december(self):
        intent = detect_attribution("为什么 2025 年 1 月销售额下降了？", ENUMS)
        self.assertEqual(intent["base_key"], (2024, 12, 12))

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


class TestQuarterIntent(unittest.TestCase):
    """季度/半年：既是"两个时段"，也是"跨度"。"""

    def test_quarter_pair_than_order(self):
        """「Q2 比 Q1 高」→ 解释 Q2，参照 Q1。"""
        intent = detect_attribution("为什么 Q2 的销售额比 Q1 高？", ENUMS)
        self.assertEqual(intent["kind"], "period")
        self.assertEqual((intent["base"], intent["current"]), ("Q1", "Q2"))
        self.assertEqual(intent["base_key"], (None, 1, 3))
        self.assertEqual(intent["current_key"], (None, 4, 6))

    def test_quarter_pair_from_to_order(self):
        intent = detect_attribution("从 Q1 到 Q2 销售额为什么上升了？", ENUMS)
        self.assertEqual((intent["base"], intent["current"]), ("Q1", "Q2"))

    def test_quarter_with_year_is_one_period_not_two(self):
        """「2025年Q2」不能被拆成「2025年」+「Q2」两个时段。"""
        intent = detect_attribution("为什么 2025 年 Q2 的销售额比 Q1 高？", ENUMS)
        self.assertEqual(intent["current_key"], (2025, 4, 6))
        self.assertEqual(intent["base_key"], (None, 1, 3))

    def test_chinese_quarter_and_half(self):
        intent = detect_attribution("为什么第二季度的销售额比第一季度高？", ENUMS)
        self.assertEqual((intent["base_key"], intent["current_key"]),
                         ((None, 1, 3), (None, 4, 6)))
        half = detect_attribution("为什么下半年的销售额比上半年低？", ENUMS)
        self.assertEqual((half["base_key"], half["current_key"]),
                         ((None, 1, 6), (None, 7, 12)))

    def test_single_quarter_defaults_to_previous_quarter(self):
        """只说一个季度时，要比的是上一季度，不是上一个月。"""
        intent = detect_attribution("为什么 2025 年 Q1 的销售额下降了？", ENUMS)
        self.assertTrue(intent["assumed"])
        self.assertEqual(intent["base_key"], (2024, 10, 12))
        self.assertEqual(intent["base"], "2024年Q4")

    def test_quarter_without_cause_word_is_not_attribution(self):
        self.assertIsNone(detect_attribution("Q2 的销售额是多少？", ENUMS))

    def test_compare_prompt_asks_to_expand_months(self):
        """提示词要说明"跨度要展开成月份"，否则模型可能只筛一个月。"""
        from app.tools.attribution import describe_comparison

        intent = detect_attribution("为什么 2025 年 Q2 的销售额比 Q1 高？", ENUMS)
        prompt = describe_comparison(intent)
        self.assertIn("跨度", prompt)
        self.assertIn("'2025-04'", prompt)     # Q2 的三行月份，提醒按月展开

    def test_month_question_gets_no_span_prompt(self):
        from app.tools.attribution import describe_comparison

        intent = detect_attribution("为什么 2025 年 6 月的销售额比 5 月高？", ENUMS)
        self.assertNotIn("跨度", describe_comparison(intent))


def quarter_crosstab() -> pd.DataFrame:
    """「月份 × 品类」交叉表，覆盖 2025 Q1（100）与 Q2（120）：

    Q1 → 手机 30 / 笔记本 70；Q2 → 手机 40 / 笔记本 80。差额 20 由两个品类各担 10。
    """
    rows = [
        ("2025-01", "手机", 10.0), ("2025-02", "手机", 20.0),
        ("2025-01", "笔记本", 40.0), ("2025-03", "笔记本", 30.0),
        ("2025-04", "手机", 15.0), ("2025-05", "手机", 25.0),
        ("2025-04", "笔记本", 45.0), ("2025-06", "笔记本", 35.0),
    ]
    return pd.DataFrame(rows, columns=["月份", "品类", "销售额"])


class TestQuarterDecompose(unittest.TestCase):
    """季度对比的拆解：对比列可能已被展开成月份，也可能直接是季度。"""

    def setUp(self):
        self.intent = detect_attribution("为什么 Q2 的销售额比 Q1 高？", ENUMS)

    def test_month_rows_are_rolled_into_quarters(self):
        att = decompose(quarter_crosstab(), self.intent)
        self.assertTrue(att["ok"])
        self.assertEqual(att["base_total"], 100.0)
        self.assertEqual(att["current_total"], 120.0)
        self.assertEqual(att["delta"], 20.0)
        self.assertEqual(att["table"].set_index("维度").loc["手机", "变化"], 10.0)
        self.assertEqual(att["table"].set_index("维度").loc["笔记本", "变化"], 10.0)

    def test_rows_outside_both_quarters_are_ignored(self):
        """7 月（Q3）的行不能被算进 Q2 —— 不然"Q2 比 Q1"会算成三个季度。"""
        frame = pd.concat([quarter_crosstab(), pd.DataFrame(
            {"月份": ["2025-07"], "品类": ["手机"], "销售额": [999.0]})], ignore_index=True)
        att = decompose(frame, self.intent)
        self.assertEqual(att["current_total"], 120.0)

    def test_quarter_column_also_works(self):
        """模型直接按季度出列（2025Q1 / 2025Q2）也要能拆。"""
        frame = pd.DataFrame({
            "季度": ["2025Q1", "2025Q1", "2025Q2", "2025Q2"],
            "品类": ["手机", "笔记本", "手机", "笔记本"],
            "销售额": [30.0, 70.0, 40.0, 80.0],
        })
        att = decompose(frame, self.intent)
        self.assertTrue(att["ok"])
        self.assertEqual(att["compare_col"], "季度")
        self.assertEqual((att["base_total"], att["current_total"]), (100.0, 120.0))
        self.assertIn("Q2", att["summary"])

    def test_quarter_span_has_no_day_count_note(self):
        """同季度对比自然天数接近，不该刷"天数不同"的提醒。"""
        att = decompose(quarter_crosstab(), self.intent)
        self.assertFalse(any("天数" in n for n in att["notes"]), att["notes"])

    def test_missing_base_quarter_says_which_side_is_missing(self):
        """只给 Q1 的数据、却比的是 2024Q4：要说明缺的是基期，而不是"找不到对比列"。"""
        frame = quarter_crosstab()
        frame = frame[frame["月份"].str.startswith(("2025-01", "2025-02", "2025-03"))]
        intent = detect_attribution("为什么 2025 年 Q1 的销售额下降了？", ENUMS)
        att = decompose(frame, intent)
        self.assertFalse(att["ok"])
        self.assertIn("2024年Q4", att["reason"])
        self.assertIn("没有可比对象", att["reason"])


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
