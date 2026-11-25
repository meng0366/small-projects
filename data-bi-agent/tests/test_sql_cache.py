"""SQL 复用（P1）的离线测试：不调模型，真库只读跑 SQL。

重点不是"缓存能不能命中"，而是**哪些情况绝不能命中**：
相对时间跨天、多轮追问、归因问题、取值待澄清、库结构变了、结果变了。
复用答错比不复用更贵 —— 用户完全看不出来这个数是旧的。
"""
import json
import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest import mock

try:
    import _env  # noqa: F401  测试环境隔离（临时 AGENT_HOME、关闭 SQL 复用）
except ModuleNotFoundError:      # 以包方式运行（python -m unittest tests.test_xxx）
    from . import _env  # noqa: F401
except ModuleNotFoundError:      # 以包方式运行（python -m unittest tests.test_xxx）
    from . import _env  # noqa: F401
import pandas as pd

import app.graph as G
from app.context import AgentContext
from app.tools import lesson_store as L
from app.tools import sql_cache as C

SALES_SQL = ("SELECT c.region AS 区域, SUM(o.amount) AS 销售额 "
             "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
             "GROUP BY c.region ORDER BY 销售额 DESC")

QUESTION = "各区域的销售额是多少？"


class CacheTestBase(unittest.TestCase):
    """每个用例一个临时 AGENT_HOME，彼此不共享缓存文件。"""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="sqlcache-")
        self.env = mock.patch.dict(os.environ, {"AGENT_HOME": self.home})
        self.env.start()
        self.ctx = AgentContext(use_cache=True)

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.home, ignore_errors=True)


class TestKey(CacheTestBase):
    def test_punctuation_and_space_do_not_matter(self):
        """问号多一个全角、少了句号，是同一个问题，不该再花一次钱。"""
        self.assertEqual(C.cache_key("各区域的销售额是多少？", self.ctx),
                         C.cache_key(" 各区域的销售额是多少 ", self.ctx))
        self.assertEqual(C.cache_key("各区域的销售额是多少？", self.ctx),
                         C.cache_key("各区域的销售额是多少", self.ctx))

    def test_different_question_different_key(self):
        self.assertNotEqual(C.cache_key(QUESTION, self.ctx),
                            C.cache_key("各品类的销售额是多少？", self.ctx))

    def test_relative_time_binds_resolved_range(self):
        """「上个月」的 SQL 里是显式日期：换一个月再复用就是拿上个月的数冒充这个月。"""
        keys = []
        for day in (date(2026, 1, 15), date(2026, 2, 10)):
            with mock.patch.object(AgentContext, "today",
                                   new_callable=mock.PropertyMock, return_value=day):
                keys.append(C.cache_key("上个月的销售额是多少？", self.ctx))
        self.assertNotEqual(keys[0], keys[1])

    def test_same_month_different_day_still_hits(self):
        """同一个月里的不同日期，「上个月」指向同一个区间，可以复用。

        绑 today 会在这里白失效一次 —— 绑"换算后的区间"才是精确的做法。
        """
        keys = []
        for day in (date(2026, 1, 15), date(2026, 1, 20)):
            with mock.patch.object(AgentContext, "today",
                                   new_callable=mock.PropertyMock, return_value=day):
                keys.append(C.cache_key("上个月的销售额是多少？", self.ctx))
        self.assertEqual(keys[0], keys[1])

    def test_absolute_question_ignores_today(self):
        """问明确年份的问题与"今天是哪天"无关，不该每天失效一次。"""
        keys = []
        for day in (date(2026, 3, 10), date(2026, 4, 10)):
            with mock.patch.object(AgentContext, "today",
                                   new_callable=mock.PropertyMock, return_value=day):
                keys.append(C.cache_key(QUESTION, self.ctx))
        self.assertEqual(keys[0], keys[1])

    def test_other_database_gets_other_key(self):
        other = AgentContext(db_file=os.path.join(self.home, "nope.db"), use_cache=True)
        self.assertNotEqual(C.cache_key(QUESTION, self.ctx),
                            C.cache_key(QUESTION, other))


class TestSkipRules(CacheTestBase):
    """哪些问题不该查缓存 —— 每条都对应一种"复用就答错"。"""

    def test_unknown_value_goes_to_clarify(self):
        entry, reason = C.find_cached_sql("办公用品类产品的销售额是多少？", self.ctx)
        self.assertIsNone(entry)
        self.assertEqual(reason, "取值待澄清")

    def test_attribution_question_is_not_cached(self):
        _entry, reason = C.find_cached_sql("6月比5月的销售额为什么下降？", self.ctx)
        self.assertEqual(reason, "归因问题")

    def test_history_blocks_lookup(self):
        _entry, reason = C.find_cached_sql(QUESTION, self.ctx, has_history=True)
        self.assertEqual(reason, "多轮追问")

    def test_no_db_has_nothing_to_reuse(self):
        csv_ctx = AgentContext(db_file=os.path.join(self.home, "x.db"), use_cache=True)
        _entry, reason = C.find_cached_sql(QUESTION, csv_ctx)
        self.assertEqual(reason, "单表模式")

    def test_switch_off(self):
        ctx = AgentContext(use_cache=False)
        _entry, reason = C.find_cached_sql(QUESTION, ctx)
        self.assertEqual(reason, "缓存已关闭")

    def test_time_out_of_range_goes_to_refusal(self):
        _entry, reason = C.find_cached_sql("下个月的销售额是多少？", self.ctx)
        self.assertEqual(reason, "时间超出覆盖")

    def test_miss_reason_is_reported(self):
        entry, reason = C.find_cached_sql(QUESTION, self.ctx)
        self.assertIsNone(entry)
        self.assertEqual(reason, "未命中")


class TestStore(CacheTestBase):
    def test_roundtrip(self):
        self.assertIsNotNone(C.remember_sql(QUESTION, SALES_SQL, self.ctx))
        entry, reason = C.find_cached_sql(QUESTION, self.ctx)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["sql"], SALES_SQL)
        self.assertIn("第 1 次", reason)

    def test_hits_accumulate(self):
        C.remember_sql(QUESTION, SALES_SQL, self.ctx)
        C.find_cached_sql(QUESTION, self.ctx)
        entry, reason = C.find_cached_sql(QUESTION, self.ctx)
        self.assertEqual(entry["hits"], 2)
        self.assertIn("第 2 次", reason)

    def test_same_question_overwrites(self):
        """同一个问题再存一次是"更新"，不是"堆积"：堆积会让淘汰算法失效。"""
        C.remember_sql(QUESTION, SALES_SQL, self.ctx)
        C.remember_sql(QUESTION, "SELECT 1", self.ctx)
        self.assertEqual(C.cache_stats(self.ctx)["entries"], 1)
        entry, _ = C.find_cached_sql(QUESTION, self.ctx)
        self.assertEqual(entry["sql"], "SELECT 1")

    def test_empty_sql_rejected(self):
        self.assertIsNone(C.remember_sql(QUESTION, "   ", self.ctx))
        self.assertEqual(C.cache_stats(self.ctx)["entries"], 0)

    def test_unsafe_sql_rejected(self):
        self.assertIsNone(C.remember_sql(QUESTION, "DROP TABLE orders", self.ctx))
        self.assertEqual(C.cache_stats(self.ctx)["entries"], 0)

    def test_attribution_not_stored(self):
        """存了也永远查不到（lookup 会跳过归因），所以干脆不存。"""
        self.assertIsNone(C.remember_sql("6月比5月的销售额为什么下降？",
                                         SALES_SQL, self.ctx))

    def test_corrupt_file_treated_as_empty(self):
        os.makedirs(C.cache_dir(), exist_ok=True)
        with open(C.cache_file(self.ctx), "w", encoding="utf-8") as f:
            f.write("{这不是 JSON")
        entry, reason = C.find_cached_sql(QUESTION, self.ctx)
        self.assertIsNone(entry)
        self.assertEqual(reason, "未命中")

    def test_tampered_sql_is_refused(self):
        """缓存文件在本地磁盘上，可以被手改；执行前必须再过一次安全检查。"""
        C.remember_sql(QUESTION, SALES_SQL, self.ctx)
        path = C.cache_file(self.ctx)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data[0]["sql"] = "DROP TABLE orders"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        entry, reason = C.find_cached_sql(QUESTION, self.ctx)
        self.assertIsNone(entry)
        self.assertEqual(reason, "条目不安全")

    def test_forget(self):
        C.remember_sql(QUESTION, SALES_SQL, self.ctx)
        self.assertTrue(C.forget_sql(QUESTION, self.ctx))
        self.assertFalse(C.forget_sql(QUESTION, self.ctx))
        self.assertEqual(C.cache_stats(self.ctx)["entries"], 0)

    def test_capacity_evicts_least_recently_used(self):
        with mock.patch.object(C, "MAX_ENTRIES", 3):
            for i in range(5):
                C.remember_sql(f"问题{i}", SALES_SQL, self.ctx)
            self.assertEqual(C.cache_stats(self.ctx)["entries"], 3)
            # 最新的三条留着，最早的被淘汰
            self.assertIsNone(C.find_cached_sql("问题0", self.ctx)[0])
            self.assertIsNotNone(C.find_cached_sql("问题4", self.ctx)[0])

    def test_teach_clears_cache(self):
        """用户修正了这条问题的口径 → 旧 SQL 不能再被复用。

        不清的话，teach 之后问题原文没变，复用直接命中修正**之前**的语句，
        于是"教过了却还是错的答案"，日志里一切正常。
        """
        C.remember_sql(QUESTION, SALES_SQL, self.ctx)
        L.add_lesson(QUESTION, "SELECT 1", ctx=self.ctx, verify=False)
        self.assertIsNone(C.find_cached_sql(QUESTION, self.ctx)[0])


class TestResultFingerprint(CacheTestBase):
    def test_same_data_same_fingerprint(self):
        a = pd.DataFrame({"区域": ["华南", "华东"], "销售额": [1.5, 2.5]})
        b = pd.DataFrame({"区域": ["华南", "华东"], "销售额": [1.5, 2.5]})
        self.assertEqual(C.result_fingerprint(a), C.result_fingerprint(b))

    def test_changed_data_changes_fingerprint(self):
        a = pd.DataFrame({"区域": ["华南"], "销售额": [1.5]})
        b = pd.DataFrame({"区域": ["华南"], "销售额": [1.6]})
        self.assertNotEqual(C.result_fingerprint(a), C.result_fingerprint(b))

    def test_empty_and_none(self):
        self.assertEqual(C.result_fingerprint(None), "")
        self.assertEqual(C.result_fingerprint(pd.DataFrame()), "empty")


class TestGraphReuse(CacheTestBase):
    """状态机侧：命中后不再生成、不再写计划、结论也可能一并复用。"""

    def _agent(self, with_insight: bool = False):
        return G.GraphAgent(self.ctx, with_chart=False, with_insight=with_insight)

    def test_second_run_skips_generation(self):
        agent = self._agent()
        with mock.patch.object(G, "generate_sql", return_value=SALES_SQL) as gen:
            first = agent.run(QUESTION)
        self.assertEqual(gen.call_count, 1)
        self.assertFalse(first.get("cache_hit"))

        with mock.patch.object(G, "generate_sql",
                               side_effect=AssertionError("命中缓存后不该再生成")) as gen2:
            second = agent.run(QUESTION)
        self.assertEqual(gen2.call_count, 0)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(len(second["result"]), len(first["result"]))

    def test_reused_insight_skips_llm(self):
        """结果一字不差时，上次的结论可以直接用 —— 这才是真正的零调用。"""
        agent = self._agent(with_insight=True)
        with mock.patch.object(G, "generate_sql", return_value=SALES_SQL), \
                mock.patch.object(G, "generate_insight", return_value="第一次的结论") as ins:
            agent.run(QUESTION)
        self.assertEqual(ins.call_count, 1)

        def boom(*a, **k):
            raise AssertionError("结果没变，不该重新生成结论")

        with mock.patch.object(G, "generate_sql", side_effect=boom), \
                mock.patch.object(G, "generate_insight", side_effect=boom):
            second = agent.run(QUESTION)
        self.assertEqual(second["insight"], "第一次的结论")

    def test_changed_result_regenerates_insight(self):
        """数据变了就复用 SQL、重算结论 —— 结论依赖数据，不能跟着 SQL 一起复读。"""
        agent = self._agent(with_insight=True)
        old = pd.DataFrame({"区域": ["华南"], "销售额": [100.0]})
        new = pd.DataFrame({"区域": ["华南"], "销售额": [200.0]})
        ok_old = {"ok": True, "df": old, "sql": SALES_SQL}
        ok_new = {"ok": True, "df": new, "sql": SALES_SQL}

        with mock.patch.object(G, "generate_sql", return_value=SALES_SQL), \
                mock.patch.object(G, "run_sql", return_value=ok_old), \
                mock.patch.object(G, "generate_insight", return_value="旧结论"):
            agent.run(QUESTION)

        with mock.patch.object(G, "run_sql", return_value=ok_new), \
                mock.patch.object(G, "generate_insight", return_value="新结论") as ins:
            second = agent.run(QUESTION)
        self.assertEqual(ins.call_count, 1)
        self.assertEqual(second["insight"], "新结论")

    def test_stale_cached_sql_is_forgotten_and_regenerated(self):
        """复用的 SQL 跑不通（表/字段变了）→ 作废条目、重新生成，且不换 pandas。

        换路是"模型理解错"的处置；这里错的是缓存本身，换成 pandas 反而绕开
        全部 JOIN 与聚合能力，得到的表也对不上。
        """
        agent = self._agent()
        with mock.patch.object(G, "generate_sql", return_value=SALES_SQL), \
                mock.patch.object(G, "run_sql", return_value={"ok": True,
                                                             "df": pd.DataFrame({"a": [1]}),
                                                             "sql": SALES_SQL}):
            agent.run(QUESTION)

        failed = {"ok": False, "stage": "执行", "error": "no such table: orders",
                  "sql": SALES_SQL}
        ok = {"ok": True, "df": pd.DataFrame({"a": [2]}), "sql": SALES_SQL}
        with mock.patch.object(G, "generate_sql", return_value=SALES_SQL) as gen, \
                mock.patch.object(G, "run_sql", side_effect=[failed, ok]):
            state = agent.run(QUESTION)

        self.assertEqual(gen.call_count, 1)          # 弃用后重新生成了
        self.assertEqual(state["route"], "sql")      # 没有换到 pandas
        self.assertIsNotNone(state["result"])
        self.assertTrue(state.get("cache_failed"))   # 明确记下了"缓存过时"这件事


class TestLinearReuse(CacheTestBase):
    """线性引擎也要能复用 —— 两种引擎行为不一致是最难解释的那种 bug。"""

    def test_linear_second_run_skips_generation(self):
        from app.agent import DataAgent

        agent = DataAgent(ctx=self.ctx, with_chart=False, with_insight=False)
        with mock.patch("app.agent.generate_sql", return_value=SALES_SQL) as gen:
            first = agent.ask(QUESTION)
        self.assertEqual(gen.call_count, 1)
        self.assertFalse(first.cache_hit)

        with mock.patch("app.agent.generate_sql",
                        side_effect=AssertionError("命中缓存后不该再生成")) as gen2:
            second = agent.ask(QUESTION)
        self.assertEqual(gen2.call_count, 0)
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(second.df), len(first.df))


if __name__ == "__main__":
    unittest.main()
