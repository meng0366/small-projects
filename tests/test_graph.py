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

        def flaky(schema, terms, question, previous_error=None):
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
