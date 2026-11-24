"""LangGraph 状态机的离线测试：全部 mock 掉 LLM，不消耗 API 额度。

覆盖：
- 正常路径（SQL 一次成功）
- SQL 失败后自动切换到 pandas
- pandas 失败后重试，且重试时能拿到上一次的真实错误（错误回喂）
- 重试次数上限与最终失败说明
"""
import unittest
from unittest import mock

import app.graph as G

GOOD_SQL = ("SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额 "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.region ORDER BY 销售额 DESC")
BAD_SQL = "SELECT nope FROM orders"
GOOD_CODE = 'result = df.groupby("region")["amount"].sum().reset_index()'
BAD_CODE = 'result = df["nope"].sum()'


class TestGraphFlow(unittest.TestCase):
    def _run(self, sql, code=None, max_retries=2):
        """构造 agent 并跑一次，sql/code 可为字符串或 side_effect。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=max_retries)
        sql_mock = mock.Mock(return_value=sql) if isinstance(sql, str) else mock.Mock(side_effect=sql)
        with mock.patch.object(G, "generate_sql", sql_mock):
            if code is None:
                with mock.patch.object(G, "generate_code", return_value=GOOD_CODE):
                    return agent.run("各区域的销售额是多少？")
            code_mock = mock.Mock(return_value=code) if isinstance(code, str) else mock.Mock(side_effect=code)
            with mock.patch.object(G, "generate_code", code_mock):
                return agent.run("各区域的销售额是多少？")

    def test_happy_path(self):
        state = self._run(GOOD_SQL)
        self.assertEqual(state["route"], "sql")
        self.assertEqual(state["retry_count"], 0)
        self.assertIsNotNone(state["result"])
        self.assertEqual(len(state["result"]), 5)

    def test_sql_failure_falls_back_to_pandas(self):
        state = self._run(BAD_SQL)
        self.assertEqual(state["route"], "pandas")     # 已换路
        self.assertEqual(state["retry_count"], 1)
        self.assertIsNotNone(state["result"])
        self.assertEqual(len(state["errors"]), 1)
        self.assertIn("SQL", state["errors"][0])

    def test_retry_receives_previous_error(self):
        """第二次生成必须带上一次的错误信息（错误回喂）。"""
        received = []

        def flaky(schema, terms, question, previous_error=None, previous_empty=False):
            received.append(previous_error)
            return BAD_CODE if len(received) == 1 else GOOD_CODE

        state = self._run(sql=RuntimeError("模拟 SQL 生成不可用"), code=flaky)
        self.assertEqual(state["retry_count"], 2)
        self.assertIsNotNone(state["result"])
        # 首次生成即失败：第二次应带上「生成失败」的原因
        self.assertIn("生成失败", received[0])
        # 第二次执行失败后：第三次应带上真实运行时报错
        self.assertIn("KeyError", received[1])

    def test_retries_exhausted_gives_readable_answer(self):
        state = self._run(BAD_SQL, code=BAD_CODE, max_retries=1)
        self.assertIsNone(state["result"])
        self.assertEqual(state["retry_count"], 1)
        self.assertIn("未能完成分析", state["answer"])
        self.assertIn("错误", state["answer"])

    def test_empty_result_triggers_one_retry(self):
        """SQL 执行成功但 0 行时，应触发一次自愈；放宽后拿到数据就正常回答。"""
        received_empty = []

        def flaky_sql(schema, terms, question, previous_error=None, previous_empty=False):
            received_empty.append(previous_empty)
            # 第一次故意给一个语法正确但必然查不到数据的 SQL，第二次给正常 SQL
            return ("SELECT * FROM orders WHERE amount < 0" if len(received_empty) == 1
                    else GOOD_SQL)

        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=2)
        with mock.patch.object(G, "generate_sql", side_effect=flaky_sql):
            state = agent.run("各区域的销售额是多少？")

        self.assertEqual(len(received_empty), 2)          # 确实重试了一次
        self.assertFalse(received_empty[0])               # 首次没有"空结果"提示
        self.assertTrue(received_empty[1])                # 重试时明确告知"上一轮为空"
        self.assertEqual(state["retry_count"], 1)
        self.assertEqual(len(state["result"]), 5)         # 放宽后拿到数据
        self.assertEqual(state["route"], "sql")           # 空结果不换路
        self.assertTrue(any("0 行" in e for e in state["errors"]))

    def test_empty_result_keeps_result_and_explains(self):
        """放宽后仍为空：要如实说明，而不是反复重试或说'分析失败'。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=2)
        with mock.patch.object(G, "generate_sql", return_value=GOOD_SQL):
            # 用一个必然返回空结果的 SQL 替身：拦住 run_sql，返回空 DataFrame
            import pandas as pd
            with mock.patch.object(G, "run_sql",
                                   return_value={"ok": True, "df": pd.DataFrame(),
                                                 "sql": "SELECT 1", "rows": 0}):
                state = agent.run("2024 年各区域的销售额是多少？")

        self.assertEqual(state["retry_count"], 1)         # 只给一次机会，不空转
        self.assertIsNotNone(state["result"])
        self.assertEqual(len(state["result"]), 0)
        self.assertEqual(state["chart_type"], "none")
        self.assertIn("没有匹配到数据", state["answer"])
        self.assertIn("放宽条件", state["answer"])

    def test_empty_result_not_retried_when_budget_zero(self):
        """max_retries=0 时应直接如实说明，不重试。"""
        import pandas as pd
        agent = G.GraphAgent(with_chart=False, with_insight=False, max_retries=0)
        with mock.patch.object(G, "generate_sql", return_value=GOOD_SQL):
            with mock.patch.object(G, "run_sql",
                                   return_value={"ok": True, "df": pd.DataFrame(),
                                                 "sql": "SELECT 1", "rows": 0}):
                state = agent.run("不存在的区域销售额是多少？")
        self.assertEqual(state["retry_count"], 0)
        self.assertIn("没有匹配到数据", state["answer"])

    def test_is_empty_result_helper(self):
        import pandas as pd
        from app.nodes.reflect import is_empty_result
        self.assertFalse(is_empty_result(None))                       # 失败不是空结果
        self.assertTrue(is_empty_result(pd.DataFrame()))              # 空表
        self.assertFalse(is_empty_result(pd.DataFrame({"a": [1]})))   # 有数据
        self.assertTrue(is_empty_result(pd.Series(dtype="float64")))  # 空序列
        self.assertFalse(is_empty_result(0))                          # 标量 0 不是空

    def test_pandas_only_mode_without_db(self):
        """数据库缺失时应直接走 pandas。"""
        agent = G.GraphAgent(with_chart=False, with_insight=False)
        agent.ctx.db_name = "not_exist.db"
        # cached_property 需重建上下文
        from app.context import AgentContext
        agent.ctx = AgentContext(db_name="not_exist.db")
        agent.graph = G.build_graph(agent.ctx, with_chart=False, with_insight=False)
        with mock.patch.object(G, "generate_code", return_value=GOOD_CODE):
            state = agent.run("各区域的销售额是多少？")
        self.assertEqual(state["route"], "pandas")
        self.assertIsNotNone(state["result"])


if __name__ == "__main__":
    unittest.main()
