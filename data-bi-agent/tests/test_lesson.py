"""修正回流（P1-1）的离线测试：不调模型、不碰真实主目录。

覆盖四类：
- 入库校验：拒绝非 SELECT、拒绝跑不通的语句、拒绝超长；同问题覆盖不堆积
- 库指纹隔离：结构不同的库不共用同一份 lesson
- 相关度挑选：字面相似但条件冲突（华南 vs 华东、区域 vs 品类、2024 vs 2025）必须不注入
- prompt 注入：lessons 出现在正确位置（通用示例之后、当前问题之前），关掉开关就一条不注入
"""
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import pandas as pd

from app.context import AgentContext
from app.nodes.generate import _build_messages
from app.tools import lesson_store as L

SQL_GOOD = ("SELECT ROUND(SUM(o.amount), 2) AS 销售额 "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.region = '华南'")


class _TempHome(unittest.TestCase):
    """所有用例都在临时 AGENT_HOME 里跑，不会碰用户真实的 ~/.data-agent。"""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="lessons-")
        self.env = mock.patch.dict(os.environ, {"AGENT_HOME": self.home})
        self.env.start()
        self.ctx = AgentContext()

    def tearDown(self):
        self.env.stop()

    def _write_db(self, path, schema_sql):
        con = sqlite3.connect(path)
        con.executescript(schema_sql)
        con.close()
        return path


class TestStore(_TempHome):
    def test_add_and_load(self):
        lesson = L.add_lesson("华南的销售额是多少？", SQL_GOOD, note="限定华南", ctx=self.ctx)
        self.assertEqual(lesson["id"], "L001")
        self.assertEqual(lesson["kind"], "sql")
        self.assertEqual(lesson["note"], "限定华南")

        items = L.load_lessons(self.ctx)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question"], "华南的销售额是多少？")

    def test_ids_increase(self):
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)
        second = L.add_lesson("每个商品的毛利率是多少？", "DEFER: 没有成本字段", ctx=self.ctx)
        self.assertEqual(second["id"], "L002")
        self.assertEqual(second["kind"], "defer")

    def test_same_question_overwrites(self):
        """同一个问题再教一次是修正，不能留下两条互相矛盾的示范。"""
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)
        fixed = SQL_GOOD.replace("华南", "华东")
        L.add_lesson("华南的销售额是多少?", fixed, ctx=self.ctx)   # 只差一个标点

        items = L.load_lessons(self.ctx)
        self.assertEqual(len(items), 1)
        self.assertIn("华东", items[0]["output"])
        self.assertIn("updated", items[0])

    def test_rejects_non_select(self):
        with self.assertRaises(L.LessonError) as ctx:
            L.add_lesson("删掉所有订单", "DELETE FROM orders", ctx=self.ctx)
        self.assertIn("只允许 SELECT", str(ctx.exception))

    def test_rejects_broken_sql(self):
        """跑不通的语句存进去就是教模型写错语句，默认必须拒绝。"""
        with self.assertRaises(L.LessonError) as ctx:
            L.add_lesson("各城市的销售额是多少？",
                         "SELECT city, SUM(amount) FROM orders GROUP BY city", ctx=self.ctx)
        self.assertIn("执行失败", str(ctx.exception))

    def test_broken_sql_allowed_with_force(self):
        lesson = L.add_lesson("各城市的销售额是多少？",
                              "SELECT city FROM orders", ctx=self.ctx, verify=False)
        self.assertEqual(lesson["kind"], "sql")

    def test_rejects_empty_and_too_long(self):
        with self.assertRaises(L.LessonError):
            L.add_lesson("   ", SQL_GOOD, ctx=self.ctx)
        with self.assertRaises(L.LessonError):
            L.add_lesson("问题", "SELECT " + "1, " * 600 + "1", ctx=self.ctx, verify=False)

    def test_defer_requires_prefix(self):
        with self.assertRaises(L.LessonError):
            L.add_lesson("毛利率是多少", "没有成本字段", kind="defer", ctx=self.ctx)

    def test_defer_skips_verification(self):
        lesson = L.add_lesson("毛利率是多少", "DEFER: products 只有标价，没有成本",
                              ctx=self.ctx, verify=True)
        self.assertEqual(lesson["kind"], "defer")

    def test_remove(self):
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)
        self.assertTrue(L.remove_lesson("L001", self.ctx))
        self.assertFalse(L.remove_lesson("L001", self.ctx))     # 已经没了
        self.assertEqual(L.load_lessons(self.ctx), [])

    def test_next_id_after_removal_does_not_collide(self):
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)
        L.add_lesson("华东的销售额是多少？", SQL_GOOD.replace("华南", "华东"), ctx=self.ctx)
        L.remove_lesson("L001", self.ctx)
        third = L.add_lesson("华北的销售额是多少？", SQL_GOOD.replace("华南", "华北"), ctx=self.ctx)
        self.assertEqual(third["id"], "L003")                   # 不重用 L001

    def test_bad_lines_are_skipped(self):
        path = L.lessons_path(self.ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"question": "坏行，缺 output"}\nnot-json\n'
                        '{"question": "好的", "output": "SELECT 1"}\n', encoding="utf-8")
        items = L.load_lessons(self.ctx)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question"], "好的")

    def test_fingerprint_isolates_databases(self):
        """结构不同的库不能共用 lesson —— 否则会教模型写出不存在的字段。"""
        first = L.db_fingerprint(self.ctx)

        other = os.path.join(self.home, "other.db")
        self._write_db(other, "CREATE TABLE weather (city TEXT, temp REAL);")
        second = L.db_fingerprint(AgentContext(db_file=other))

        self.assertNotEqual(first, second)
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)
        self.assertEqual(len(L.load_lessons(self.ctx)), 1)
        self.assertEqual(len(L.load_lessons(AgentContext(db_file=other))), 0)


class TestScoring(_TempHome):
    def setUp(self):
        super().setUp()
        self.enums = self.ctx.enums
        self.region_lesson = {"question": "华南的销售额是多少？"}
        self.margin_lesson = {"question": "每个商品的毛利率是多少？"}

    def _score(self, question, lesson):
        return L.score_lesson(question, lesson, self.enums)

    def test_same_topic_scores_high(self):
        self.assertGreater(self._score("华南地区的销售额", self.region_lesson), 0.42)

    def test_value_conflict_is_zero(self):
        """华南的示范不能套到华东的问题上。"""
        self.assertEqual(self._score("华东的销售额是多少？", self.region_lesson), 0.0)

    def test_dimension_conflict_is_zero(self):
        """区域 vs 品类：共享"的销售额"三个字也不该算同类。"""
        self.assertEqual(self._score("各品类的销售额是多少？", self.region_lesson), 0.0)

    def test_year_conflict_is_zero(self):
        lesson = {"question": "2024 年各区域的销售额"}
        self.assertEqual(self._score("2025 年各区域的销售额", lesson), 0.0)

    def test_generic_question_does_not_borrow_specific_lesson(self):
        """泛指"各区域"不该拿到"华南"的示范（会把范围缩小）。"""
        self.assertLess(self._score("各区域的销售额是多少？", self.region_lesson), 0.42)

    def test_unrelated_is_below_threshold(self):
        """共享"每个"两个字会有零点几的偶然重合，但必须落在门槛之下。"""
        self.assertLess(self._score("每个月的销售额走势如何？", self.margin_lesson),
                        L.MIN_SCORE)

    def test_select_respects_limit_and_threshold(self):
        lessons = [
            {"id": "L001", "question": "华南的销售额是多少？", "output": SQL_GOOD,
             "created": "2026-11-24T10:00:00"},
            {"id": "L002", "question": "华南地区的销售额是多少？", "output": SQL_GOOD,
             "created": "2026-11-24T11:00:00"},
            {"id": "L003", "question": "每个月的销售额走势如何？", "output": "SELECT 1",
             "created": "2026-11-24T12:00:00"},
        ]
        picked = L.select_lessons("华南的销售额是多少", lessons, self.enums)
        self.assertTrue(picked)
        self.assertTrue(all(item["id"] != "L003" for item in picked))   # 无关的不注入
        self.assertLessEqual(len(picked), L.MAX_LESSONS_IN_PROMPT)

    def test_budget_drops_extra_lessons(self):
        """超出注入预算时少给几条，而不是截断某一条（截断的 SQL 会教坏模型）。"""
        long_sql = "SELECT " + "a, " * 300 + "a FROM t"
        lessons = [{"id": f"L{i:03d}", "question": f"华南的销售额是多少{i}？",
                    "output": long_sql, "created": f"2026-11-24T1{i}:00:00"} for i in range(1, 4)]
        picked = L.select_lessons("华南的销售额是多少", lessons, self.enums, limit=3)

        self.assertLess(len(picked), len(lessons))          # 确实丢了多余的
        cost = sum(len(item["output"]) + len(item["question"]) for item in picked)
        self.assertLessEqual(cost, L.MAX_INJECT_CHARS)      # 但没有超预算
        for item in picked:
            self.assertEqual(item["output"], long_sql)      # 每条都是完整的，没被截断

    def test_empty_inputs(self):
        self.assertEqual(L.select_lessons("", [self.region_lesson], self.enums), [])
        self.assertEqual(L.select_lessons("华南的销售额", [], self.enums), [])


class TestInjection(_TempHome):
    """验证 lesson 真的进了 prompt，且位置正确。"""

    def _messages(self, question, lessons):
        return _build_messages("SYS", [("示例问题", "示例输出")], "{schema}{terms}问题：{question}",
                               "SCHEMA", "TERMS", question, lessons=lessons)

    def test_lesson_appended_after_examples_before_question(self):
        lesson = {"question": "华南的销售额是多少？", "output": SQL_GOOD}
        messages = self._messages("华南地区卖了多少钱", [lesson])

        contents = [m.content for m in messages]
        self.assertEqual(len(messages), 1 + 2 + 2 + 1)   # system + 示例一对 + lesson 一对 + 当前问题
        self.assertIn(SQL_GOOD, contents)
        # lesson 的答案必须排在通用示例之后、当前问题之前
        self.assertLess(contents.index(SQL_GOOD), len(contents) - 1)
        self.assertIn("华南地区卖了多少钱", contents[-1])

    def test_no_lessons_keeps_old_shape(self):
        messages = self._messages("华南地区卖了多少钱", [])
        self.assertEqual(len(messages), 4)          # system + 示例一对 + 当前问题

    def test_context_switch_disables_lessons(self):
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)
        self.assertEqual(len(self.ctx.lessons), 1)

        off = AgentContext(use_lessons=False)
        self.assertEqual(off.lessons, [])           # 评测基线用这个口径


class TestEndToEnd(_TempHome):
    """走一遍状态机：lesson 真正影响了发给模型的消息。"""

    def test_generate_node_injects_selected_lesson(self):
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)

        captured = {}

        def fake_generate_sql(schema, terms, question, previous_error=None,
                              previous_empty=False, lessons=None, plan_hint=""):
            captured["lessons"] = lessons
            return SQL_GOOD

        import app.graph as G
        agent = G.GraphAgent(ctx=self.ctx, with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_sql", side_effect=fake_generate_sql):
            with mock.patch.object(G, "run_sql",
                                   return_value={"ok": True, "df": pd.DataFrame({"销售额": [1.0]}),
                                                 "sql": SQL_GOOD, "rows": 1}):
                state = agent.run("华南地区的销售额是多少？")

        self.assertEqual([item["id"] for item in captured["lessons"]], ["L001"])
        self.assertEqual(state["lessons_used"], ["L001"])

    def test_unrelated_question_injects_nothing(self):
        L.add_lesson("华南的销售额是多少？", SQL_GOOD, ctx=self.ctx)

        captured = {}

        def fake_generate_sql(schema, terms, question, previous_error=None,
                              previous_empty=False, lessons=None, plan_hint=""):
            captured["lessons"] = lessons
            return SQL_GOOD

        import app.graph as G
        agent = G.GraphAgent(ctx=self.ctx, with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_sql", side_effect=fake_generate_sql):
            with mock.patch.object(G, "run_sql",
                                   return_value={"ok": True, "df": pd.DataFrame({"销售额": [1.0]}),
                                                 "sql": SQL_GOOD, "rows": 1}):
                state = agent.run("每个月的销售额走势如何？")

        self.assertEqual(captured["lessons"], [])
        self.assertEqual(state["lessons_used"], [])


if __name__ == "__main__":
    unittest.main()
