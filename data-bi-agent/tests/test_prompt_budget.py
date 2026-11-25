"""提示词预算（0.14.0）：schema / 术语在整份 prompt 里只出现一次。

背景：schema 与术语原来是 `user_template` 的占位符，于是每条 few-shot 示例
都带一份。实测示例库上 schema 1299 字符 + 术语 198 字符，重复 4 次（3 条示例
+ 当前问题），在 7313 字符的 prompt 里占了 5988 —— **82% 是纯重复**。

为什么必须有用例盯着：这类退化是"改模板的人看不见"的 ——
把 `{schema}` 写回模板，功能完全正常、用例（除本节）全绿，只是每次调用
多花一倍 token。所以断言的对象不是"文本里有 schema"，而是**出现次数**。

同时断言 `_build_messages` 的签名与消息条数不变：
lesson 的位置、重试反馈的位置都依赖这个结构，改结构会连带改坏它们。
"""
import unittest
from unittest import mock

from app.nodes.generate import _build_messages
from app.prompts.templates import (
    FEW_SHOT_ATTRIBUTION,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_SQL,
    SQL_USER_TEMPLATE,
    SYSTEM_ANALYST,
    SYSTEM_SQL,
    USER_TEMPLATE,
)

SCHEMA = "表 orders，3877 行\n  - amount FLOAT 金额\n  - order_date DATE 下单日期"
TERMS = "业务术语口径：\n- 销售额：SUM(orders.amount)"


class TestSchemaInjectedOnce(unittest.TestCase):
    """schema / 术语只出现一次，且落在 system 里。"""

    def _messages(self, examples, system=SYSTEM_SQL, template=SQL_USER_TEMPLATE,
                  question="各区域的销售额是多少？", **kwargs):
        return _build_messages(system, examples, template, SCHEMA, TERMS, question, **kwargs)

    def test_schema_appears_once_in_whole_prompt(self):
        messages = self._messages(FEW_SHOT_SQL)
        total = sum(m.content.count(SCHEMA) for m in messages)
        self.assertEqual(total, 1, "schema 只该出现一次，否则每条示例都在重复注入")

    def test_schema_lives_in_system_message(self):
        messages = self._messages(FEW_SHOT_SQL)
        self.assertIn(SCHEMA, messages[0].content)
        for m in messages[1:]:
            self.assertNotIn(SCHEMA, m.content)

    def test_terms_appear_once(self):
        messages = self._messages(FEW_SHOT_SQL)
        # 用带冒号的标题计数：SYSTEM_SQL 正文里也提到"业务术语口径"四个字，
        # 那是给模型的说明，不是注入的词典内容，不能算进去。
        total = sum(m.content.count("业务术语口径：") for m in messages)
        self.assertEqual(total, 1)

    def test_attribution_examples_also_deduped(self):
        messages = self._messages(FEW_SHOT_ATTRIBUTION)
        self.assertEqual(sum(m.content.count(SCHEMA) for m in messages), 1)

    def test_pandas_path_deduped(self):
        messages = self._messages(FEW_SHOT_EXAMPLES, system=SYSTEM_ANALYST,
                                  template=USER_TEMPLATE)
        self.assertEqual(sum(m.content.count(SCHEMA) for m in messages), 1)


class TestMessageShapeUnchanged(unittest.TestCase):
    """结构不能变：示例成对、当前问题在末尾、lesson 与反馈追加在最后。"""

    def test_sql_shape(self):
        messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                                   SCHEMA, TERMS, "各区域的销售额是多少？")
        self.assertEqual(len(messages), 1 + 2 * len(FEW_SHOT_SQL) + 1)
        self.assertIn("各区域的销售额是多少？", messages[-1].content)

    def test_question_still_carried_in_examples(self):
        messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                                   SCHEMA, TERMS, "问题X")
        first_example_question = FEW_SHOT_SQL[0][0]
        self.assertIn(first_example_question, messages[1].content)

    def test_plan_hint_attached_to_current_question_only(self):
        messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                                   SCHEMA, TERMS, "问题X", plan_hint="【计划】按区域分组")
        self.assertIn("【计划】按区域分组", messages[-1].content)
        for m in messages[1:-1]:
            self.assertNotIn("【计划】按区域分组", m.content)

    def test_error_feedback_still_appended(self):
        messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                                   SCHEMA, TERMS, "问题X", previous_error="no such column")
        self.assertIn("no such column", messages[-1].content)
        self.assertEqual(len(messages), 1 + 2 * len(FEW_SHOT_SQL) + 2)


class TestBudgetActuallyShrinks(unittest.TestCase):
    """拿"老写法"当基准，断言确实显著变短 —— 否则这套改动就没有意义。

    基准必须用**接近真实尺寸**的 schema：实际 schema 是 1299 字符，若用几十字符的
    玩具 schema，固定成本（SYSTEM_SQL 1221 字符）会盖过重复量，比例失真。
    """

    # 约 1300 字符，与示例库的真实 schema（1299）同量级
    BIG_SCHEMA = (SCHEMA + "\n") * 15
    BIG_TERMS = (TERMS + "\n") * 8

    def _old_style_total(self, examples, question="各区域的销售额是多少？"):
        """复现 0.13.1 的写法：每条消息都带整份 schema + 术语。

        必须把示例的**答案**也算进来（AIMessage 的 SQL 同样是 prompt 的一部分）——
        一开始用算术近似只算了 user 消息，差了 485 字符，正是三条示例答案的长度。
        """
        old_template = ("{schema}\n\n{terms}\n\n用户问题：{question}\n\n"
                        "请只输出一条 SQL 语句。")
        total = len(SYSTEM_SQL)
        for q, output in examples:
            total += len(old_template.format(schema=self.BIG_SCHEMA,
                                             terms=self.BIG_TERMS, question=q))
            total += len(output)
        total += len(old_template.format(schema=self.BIG_SCHEMA,
                                         terms=self.BIG_TERMS, question=question))
        return total

    def test_shrinks_by_at_least_40_percent(self):
        messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                                   self.BIG_SCHEMA, self.BIG_TERMS, "各区域的销售额是多少？")
        new = sum(len(m.content) for m in messages)
        old = self._old_style_total(FEW_SHOT_SQL)
        self.assertLess(new, old * 0.6, f"新写法 {new} 字符，老写法 {old} 字符，优化幅度不足 40%")

    def test_saving_equals_removed_duplicates(self):
        """省下的量 = 每条示例少带的那份 schema+术语（3 条），再扣掉 system 里新增的标题。"""
        messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                                   self.BIG_SCHEMA, self.BIG_TERMS, "各区域的销售额是多少？")
        new = sum(len(m.content) for m in messages)
        old = self._old_style_total(FEW_SHOT_SQL)
        removed = (len(self.BIG_SCHEMA) + len(self.BIG_TERMS)) * len(FEW_SHOT_SQL)
        # 余量给 system 里新增的引导语与分隔空行（约 30 字符）
        self.assertAlmostEqual(old - new, removed, delta=100)

    def test_templates_no_longer_reference_schema(self):
        """模板里不该再有 {schema} / {terms}：写回去就等于每条示例重复一份。"""
        for name, tpl in (("SQL_USER_TEMPLATE", SQL_USER_TEMPLATE),
                          ("USER_TEMPLATE", USER_TEMPLATE)):
            self.assertNotIn("{schema}", tpl, f"{name} 又带上了 schema 占位符")
            self.assertNotIn("{terms}", tpl, f"{name} 又带上了 terms 占位符")


class TestBackwardCompatibility(unittest.TestCase):
    """老调用方式（模板里仍写 {schema}{terms}）不能报错。"""

    def test_template_with_placeholders_still_formats(self):
        messages = _build_messages("SYS", [("示例问题", "示例输出")],
                                   "{schema}{terms}问题：{question}",
                                   SCHEMA, TERMS, "问题X")
        self.assertIn("示例输出", [m.content for m in messages])
        self.assertIn("问题X", messages[-1].content)

    def test_empty_schema_and_terms(self):
        messages = _build_messages(SYSTEM_SQL, FEW_SHOT_SQL, SQL_USER_TEMPLATE,
                                   "", "", "问题X")
        # 没有 schema / 术语时不该留下空标题
        self.assertNotIn("唯一权威来源", messages[0].content)
        self.assertEqual(len(messages), 1 + 2 * len(FEW_SHOT_SQL) + 1)


class TestGenerateUsesSystemContext(unittest.TestCase):
    """端到端确认：generate_sql 发出的请求里 schema 只有一份。"""

    def test_generate_sql_sends_schema_once(self):
        captured = {}

        class _Resp:
            content = "SELECT 1"

        class _LLM:
            def invoke(self, messages):
                captured["messages"] = messages
                return _Resp()

        with mock.patch("app.nodes.generate.get_llm", return_value=_LLM()):
            from app.nodes.generate import generate_sql
            generate_sql(SCHEMA, TERMS, "各区域的销售额是多少？")

        messages = captured["messages"]
        self.assertEqual(sum(m.content.count(SCHEMA) for m in messages), 1)


if __name__ == "__main__":
    unittest.main()
