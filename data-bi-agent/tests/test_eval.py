"""评测模块的离线测试：不调用 LLM，验证评测集与判分逻辑。

覆盖：
- 评测集本身自洽（id 唯一、字段完整、覆盖各分类）
- 金标准 SQL 全部可执行（评测集不会因为写错 SQL 而"凭空判错"）
- 三种比对模式（scalar / text / table）的正例与反例
- 该拒答 / 该澄清（refuse / clarify）的正例与反例：给了数就判错
- 列名、列顺序、量纲不影响判分；值算错必须判错
"""
import unittest

import pandas as pd

from evals.judge import judge, norm_cell, to_matrix
from evals.run_eval import (ABSTAIN_MODES, ANSWERABLE_MODES, CASES_PATH,
                            load_cases, run_gold)
from app.context import AgentContext

ANSWERABLE = ANSWERABLE_MODES


class TestCases(unittest.TestCase):
    """评测集自身的质量检查。"""

    def setUp(self):
        self.cases = load_cases()

    def test_case_count_and_coverage(self):
        self.assertGreaterEqual(len(self.cases), 30, "评测集至少 30 条才够看准确率")
        categories = {c["category"] for c in self.cases}
        for expected in ("总量指标", "维度汇总", "排名 Top-N", "条件筛选", "业务口径",
                         "时间趋势", "多轮追问", "该拒答", "该澄清"):
            self.assertIn(expected, categories)

    def test_ids_unique_and_fields_complete(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), "用例 id 不能重复")
        for case in self.cases:
            self.assertIn(case["mode"], ANSWERABLE + ABSTAIN_MODES)
            self.assertTrue(case["question"].strip())
            if case["mode"] in ANSWERABLE:
                self.assertTrue(case["gold_sql"].strip(),
                                f"{case['id']} 缺少金标准 SQL")

    def test_abstain_cases_have_expectation(self):
        """拒答/澄清没有金标准，就必须把"期望行为"写清楚，否则失败了没法复盘。"""
        abstain = [c for c in self.cases if c["mode"] in ABSTAIN_MODES]
        self.assertGreaterEqual(len(abstain), 6, "该拒答/该澄清用例太少，覆盖不住盲区")
        for case in abstain:
            self.assertFalse(case.get("gold_sql"),
                             f"{case['id']} 是{case['mode']}用例，不该有金标准答案")
            self.assertTrue(case.get("note"), f"{case['id']} 缺少 note（为什么值得测）")
            if case["mode"] == "clarify":
                self.assertTrue(case.get("expect_mention"), f"{case['id']} 缺少 expect_mention")
                self.assertGreaterEqual(len(case.get("expect_values") or []), 2,
                                        f"{case['id']} 的 expect_values 太少")

    def test_clarify_expect_values_are_real(self):
        """澄清用例里列的"真实可选值"必须真的在库里，否则等于教模型说错话。"""
        real = set()
        for enum in AgentContext().data_profile["enums"]:
            real.update(str(v) for v in enum.get("values") or [])
        for case in self.cases:
            if case["mode"] != "clarify":
                continue
            for value in case["expect_values"]:
                self.assertIn(str(value), real, f"{case['id']} 列的取值 {value} 不在库里")

    def test_gold_sql_is_readonly(self):
        for case in self.cases:
            if not case.get("gold_sql"):
                continue
            head = case["gold_sql"].strip().lower()
            self.assertTrue(head.startswith(("select", "with")),
                            f"{case['id']} 的金标准必须是查询语句")

    def test_gold_sql_all_executable(self):
        """金标准跑不通就等于评测集坏了，这里一次性全跑一遍（本地 SQLite，很快）。"""
        db_path = AgentContext().db_path
        for case in self.cases:
            if not case.get("gold_sql"):
                continue
            frames = run_gold(case, db_path)
            self.assertTrue(frames)
            self.assertFalse(frames[0].empty, f"{case['id']} 的标准答案为空")

    def test_dry_run_entry_exists(self):
        # 防止有人只改 cases 忘了改脚本：yaml 路径必须真实存在
        self.assertTrue(CASES_PATH.exists())


class TestNormCell(unittest.TestCase):
    def test_numbers_and_text(self):
        self.assertEqual(norm_cell(12), 12.0)
        self.assertEqual(norm_cell("1,234.5"), 1234.5)
        self.assertEqual(norm_cell("¥ 99.00"), 99.0)
        self.assertIsNone(norm_cell(None))
        self.assertIsNone(norm_cell(float("nan")))
        self.assertEqual(norm_cell(" 华东 "), "华东")
        # 月份这类字符串不能被误转成数字
        self.assertEqual(norm_cell("2025-01"), "2025-01")


class TestScalarJudge(unittest.TestCase):
    def _gold(self, value):
        return [pd.DataFrame({"x": [value]})]

    def test_match(self):
        got = pd.DataFrame({"总销售额": [17939893.45]})
        res = judge(got, self._gold(17939893.45), {"mode": "scalar"})
        self.assertTrue(res.matched)

    def test_match_within_tolerance(self):
        got = pd.DataFrame({"销售额": [17939893.4512]})
        res = judge(got, self._gold(17939893.45), {"mode": "scalar"})
        self.assertTrue(res.matched, "分位以下的浮点差异不应判错")

    def test_wrong_value(self):
        got = pd.DataFrame({"销售额": [17000000.0]})
        res = judge(got, self._gold(17939893.45), {"mode": "scalar"})
        self.assertFalse(res.matched)
        self.assertIn("17,000,000", res.detail.replace(",", ","))

    def test_multi_row_rejected(self):
        got = pd.DataFrame({"区域": ["华东", "华南"], "销售额": [993052.95, 5064245.4]})
        res = judge(got, self._gold(993052.95), {"mode": "scalar"})
        self.assertFalse(res.matched, "汇总题返回明细表应判错")

    def test_dimension_plus_metric_row(self):
        # {"区域": "华东", "销售额": 993052.95} 一行两列也应算对
        got = pd.DataFrame({"区域": ["华东"], "销售额": [993052.95]})
        res = judge(got, self._gold(993052.95), {"mode": "scalar"})
        self.assertTrue(res.matched)

    def test_empty_result(self):
        res = judge(None, self._gold(1), {"mode": "scalar"})
        self.assertFalse(res.matched)
        self.assertIn("没有返回结果", res.detail)


class TestTextJudge(unittest.TestCase):
    def test_first_row_hit(self):
        got = pd.DataFrame({"品类": ["手机"], "销售额": [3672668.0]})
        res = judge(got, [pd.DataFrame({"x": ["手机"]})], {"mode": "text"})
        self.assertTrue(res.matched)

    def test_top5_ranking_first_row_is_answer(self):
        got = pd.DataFrame({"品类": ["手机", "笔记本"], "销售额": [3672668.0, 3000000.0]})
        res = judge(got, [pd.DataFrame({"x": ["手机"]})], {"mode": "text"})
        self.assertTrue(res.matched, "降序排行第一行就是答案，应判对")

    def test_wrong_dimension(self):
        got = pd.DataFrame({"品类": ["平板"]})
        res = judge(got, [pd.DataFrame({"x": ["手机"]})], {"mode": "text"})
        self.assertFalse(res.matched)


class TestTableJudge(unittest.TestCase):
    def setUp(self):
        self.gold = [pd.DataFrame({
            "区域": ["华东", "华南", "华北"],
            "销售额": [993052.95, 5064245.40, 4000000.00],
        })]

    def test_exact_match(self):
        res = judge(self.gold[0].copy(), self.gold, {"mode": "table"})
        self.assertTrue(res.matched)
        self.assertEqual(res.mode, "strict")

    def test_column_order_and_name_ignored(self):
        got = pd.DataFrame({
            "total": [5064245.40, 993052.95, 4000000.00],
            "region": ["华南", "华东", "华北"],
        })
        res = judge(got, self.gold, {"mode": "table"})
        self.assertTrue(res.matched, "列名与列顺序不应影响判分")
        self.assertEqual(res.mode, "lenient")

    def test_row_order_ignored(self):
        got = self.gold[0].iloc[::-1].reset_index(drop=True)
        res = judge(got, self.gold, {"mode": "table"})
        self.assertTrue(res.matched)

    def test_wrong_value_rejected(self):
        got = self.gold[0].copy()
        got.loc[0, "销售额"] = 993052.96 + 1
        res = judge(got, self.gold, {"mode": "table"})
        self.assertFalse(res.matched)

    def test_row_count_mismatch_rejected(self):
        got = self.gold[0].head(2)
        res = judge(got, self.gold, {"mode": "table"})
        self.assertFalse(res.matched)
        self.assertIn("行数不一致", res.detail)

    def test_percent_scale_tolerated(self):
        got = pd.DataFrame({"区域": ["华东", "华南"], "占比": [53.6, 46.4]})
        gold = [pd.DataFrame({"区域": ["华东", "华南"], "占比": [0.536, 0.464]})]
        res = judge(got, gold, {"mode": "table"})
        self.assertTrue(res.matched, "占比写成百分比或小数都应算对")


class TestRefuseJudge(unittest.TestCase):
    """该拒答：只要给了数就判错，文案好不好不算错。"""

    spec = {"mode": "refuse"}

    def test_pass_when_explains(self):
        state = {"answer": "查询已执行成功，但当前条件下没有匹配到数据。"}
        res = judge(None, [], self.spec, state=state)
        self.assertTrue(res.matched)
        self.assertEqual(res.mode, "refuse")

    def test_aggregate_null_counts_as_no_data(self):
        # SUM 无匹配返回 1 行 NULL，行数不为 0，但不该算"给了数"
        got = pd.DataFrame({"总销售额": [None]})
        state = {"answer": "当前条件下没有匹配到数据，库里覆盖 2025 年。"}
        res = judge(got, [], self.spec, state=state)
        self.assertTrue(res.matched)

    def test_fail_when_numbers_given(self):
        got = pd.DataFrame({"月份": ["2025-03"], "销售额": [1234.0]})
        state = {"answer": "共返回 1 行结果。"}
        res = judge(got, [], self.spec, state=state)
        self.assertFalse(res.matched)
        self.assertIn("本应说明答不了", res.detail)

    def test_fail_when_silent(self):
        res = judge(None, [], self.spec, state={"answer": ""})
        self.assertFalse(res.matched)
        self.assertIn("没有说明", res.detail)

    def test_fail_when_answer_has_no_reason(self):
        res = judge(None, [], self.spec, state={"answer": "好的，已为你查询。"})
        self.assertFalse(res.matched)

    def test_error_ending_passes_but_noted(self):
        # 没编数但以报错收场：不判错（本质是诚实），但要在 detail 里留痕
        res = judge(None, [], self.spec, state={"answer": "未能完成分析。最后一次错误：no such column"})
        self.assertTrue(res.matched)
        self.assertIn("执行失败", res.detail)


class TestClarifyJudge(unittest.TestCase):
    """该澄清：要点明哪个值不存在、列出可选值，且不能给数字。"""

    spec = {"mode": "clarify", "expect_mention": "办公用品",
            "expect_values": ["平板", "手机", "配件"]}
    good = ("你提到的「办公用品」不在库里的商品品类取值中。\n"
            "现有可选值：平板 / 手机 / 配件")

    def test_pass(self):
        res = judge(None, [], self.spec, state={"answer": self.good, "unknown_enum": {"column": "category"}})
        self.assertTrue(res.matched)
        self.assertIn("取值校验", res.detail)

    def test_fail_when_mapped_to_similar_value(self):
        # 这正是"办公用品→配件给数"的历史事故
        got = pd.DataFrame({"品类": ["配件"], "销售额": [395283.75]})
        res = judge(got, [], self.spec, state={"answer": "配件的销售额是 395,283.75"})
        self.assertFalse(res.matched)

    def test_fail_when_candidate_not_named(self):
        state = {"answer": "这个品类不在库里。可选值：平板 / 手机 / 配件"}
        res = judge(None, [], self.spec, state=state)
        self.assertFalse(res.matched)
        self.assertIn("办公用品", res.detail)

    def test_fail_when_no_options(self):
        state = {"answer": "你提到的「办公用品」不在库里的品类取值中。"}
        res = judge(None, [], self.spec, state=state)
        self.assertFalse(res.matched)
        self.assertIn("可选值", res.detail)


class TestMatrix(unittest.TestCase):
    def test_to_matrix_keeps_shape(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        self.assertEqual(to_matrix(df), [[1.0, "x"], [2.0, "y"]])


if __name__ == "__main__":
    unittest.main()
