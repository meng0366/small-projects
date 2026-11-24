"""跑评测集，输出准确率报告（对应 S5.1 建集 + S5.2 量化评测）。

用法：
  # 1. 先校验评测集本身（只跑金标准 SQL，不花钱）
  python -m evals.run_eval --dry-run

  # 2. 跑全量评测（会真实调用模型，产生费用）
  python -m evals.run_eval

  # 3. 只跑几条快速验证
  python -m evals.run_eval --only A01,B01,G01

  # 4. 指定分类 / 条数
  python -m evals.run_eval --category 多轮追问 --limit 3

  # 5. 连图表与洞察一起测（更接近真实体验，但更慢更贵）
  python -m evals.run_eval --with-insight

输出：
- 控制台：逐条结果 + 汇总指标
- evals/reports/eval-时间戳.md   人读的报告
- evals/reports/eval-时间戳.json 机器读的明细（便于多次运行对比）

指标口径：
- 准确率      = 判分通过 / 全部用例（分母是全部，不是"成功的那些"，避免失败被藏起来）
- 执行成功率  = 拿到非空结果 / 全部用例
- 一次通过率  = 首次生成即正确 / 全部用例（retry_count == 0 且判分通过）
- 自愈修复率  = 重试后最终成功 / 发生过重试的用例
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from app.config import LLM_MODEL  # noqa: E402
from app.context import AgentContext  # noqa: E402
from app.graph import GraphAgent  # noqa: E402
from app.logging_utils import TokenCounter  # noqa: E402
from app.tools.sql_tool import run_sql  # noqa: E402
from evals.judge import judge, to_matrix  # noqa: E402

CASES_PATH = ROOT / "evals" / "cases.yaml"
REPORT_DIR = ROOT / "evals" / "reports"

# 有金标准答案、比"数对不对"的模式；以及比"该不该给数"的模式
ANSWERABLE_MODES = ("scalar", "text", "table")
ABSTAIN_MODES = ("refuse", "clarify")
VALID_MODES = ANSWERABLE_MODES + ABSTAIN_MODES

# Windows 控制台默认 GBK，强制 UTF-8 避免中文与符号打印报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_cases() -> list:
    """读取评测集并做基本校验。

    refuse / clarify 两类用例没有金标准 SQL —— 它们判的是"该不该给数"，
    不是"数对不对"，所以只校验模式相关字段。
    """
    data = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    if not cases:
        raise SystemExit(f"评测集为空：{CASES_PATH}")

    seen = set()
    for case in cases:
        cid = case.get("id")
        if not cid:
            raise SystemExit(f"用例缺少 id：{case.get('question')}")
        if cid in seen:
            raise SystemExit(f"用例 id 重复：{cid}")
        seen.add(cid)
        if not case.get("question"):
            raise SystemExit(f"用例 {cid} 缺少 question")

        mode = case.get("mode")
        if mode not in VALID_MODES:
            raise SystemExit(f"用例 {cid} 的 mode 非法：{mode}")
        if mode in ANSWERABLE_MODES and not case.get("gold_sql"):
            raise SystemExit(f"用例 {cid} 缺少 gold_sql")
        if mode == "clarify" and not case.get("expect_values"):
            raise SystemExit(f"用例 {cid} 是澄清题，必须给出 expect_values（真实可选值）")
    return cases


def run_gold(case: dict, db_path: str) -> list:
    """执行金标准 SQL，返回标准答案 DataFrame 列表（首个为主口径，其余为等价口径）。

    该拒答 / 该澄清的用例没有金标准，返回空列表。
    """
    if not case.get("gold_sql"):
        return []
    sqls = [case["gold_sql"]] + list(case.get("alt_sql") or [])
    frames = []
    for sql in sqls:
        res = run_sql(sql, db_path)
        if not res["ok"]:
            raise SystemExit(f"用例 {case['id']} 的金标准 SQL 执行失败：{res['error']}")
        frames.append(res["df"])
    return frames


def select_cases(cases: list, args) -> list:
    """按 --only / --category / --limit 过滤。"""
    selected = cases
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        missing = set(wanted) - {c["id"] for c in cases}
        if missing:
            raise SystemExit(f"找不到用例：{', '.join(sorted(missing))}")
        selected = [c for c in selected if c["id"] in wanted]
    if args.category:
        selected = [c for c in selected if c["category"] == args.category]
        if not selected:
            raise SystemExit(f"分类「{args.category}」下没有用例")
    if args.limit:
        selected = selected[: args.limit]
    return selected


def dry_run(cases: list, db_path: str) -> None:
    """只跑金标准 SQL，确认评测集自洽（不调用模型、不产生费用）。"""
    print(f"校验评测集：{len(cases)} 条用例，数据库 {db_path}\n")
    for i, case in enumerate(cases, 1):
        if case["mode"] in ABSTAIN_MODES:
            what = "澄清并列出可选值" if case["mode"] == "clarify" else "如实说明答不了"
            print(f"[{i:2d}/{len(cases)}] {case['id']} {case['mode']:7s} 期望：{what}（无金标准）")
            continue
        frames = run_gold(case, db_path)
        gold = frames[0]
        matrix = to_matrix(gold)
        preview = matrix[0] if matrix else []
        head = ", ".join(str(v) for v in preview[:4])
        extra = f"（{len(matrix)} 行）" if len(matrix) > 1 else ""
        alts = f" +{len(frames) - 1} 个等价口径" if len(frames) > 1 else ""
        print(f"[{i:2d}/{len(cases)}] {case['id']} {case['mode']:7s} {head}{extra}{alts}")
    print("\n评测集校验通过，可以正式运行：python -m evals.run_eval")


def percentile(values: list, ratio: float) -> float:
    """取分位数（线性插值，样本量小时也稳定）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * ratio
    low, high = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def expected_preview(case: dict, golds: list) -> str:
    """失败复盘用的"期望"文本：有金标准就摆数值，没有就摆行为要求。"""
    if golds:
        return ", ".join(str(v) for v in to_matrix(golds[0])[0][:4])
    if case["mode"] == "clarify":
        candidate = case.get("expect_mention") or "该取值"
        values = " / ".join(str(v) for v in (case.get("expect_values") or []))
        return f"说明「{candidate}」不在库里，并列出：{values}"
    return "说明这段数据/字段不存在，不给数字"


def run_one(agent: GraphAgent, case: dict, golds: list, with_insight: bool) -> dict:
    """执行一条用例，返回明细记录。"""
    counter = TokenCounter()
    started = time.perf_counter()
    error_text = ""
    state: dict = {}
    try:
        state = agent.run(
            case["question"],
            history=case.get("history") or [],
            config={"callbacks": [counter]},
        )
    except Exception as e:  # 评测不能因为单条崩溃中断
        error_text = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - started

    result = state.get("result")
    if error_text:
        state = {**state, "answer": state.get("answer") or error_text}
    verdict = judge(result, golds or [], case, state=state)

    actual_preview = ""
    if result is not None and not getattr(result, "empty", True):
        try:
            actual_preview = ", ".join(str(v) for v in to_matrix(result)[0][:4])
        except Exception:
            actual_preview = str(result)[:120]

    abstain = case["mode"] in ABSTAIN_MODES
    # 该拒答/该澄清的用例本来就不该有结果，成功与否看判分结论，而不是"有没有拿到数"
    success = bool(verdict.matched) if abstain else (
        result is not None and not getattr(result, "empty", True))

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "mode": case.get("mode"),
        "route": state.get("route", ""),
        "clarify": bool(state.get("unknown_enum")),
        "rewritten": state.get("rewritten_question", ""),
        "statement": (state.get("statement") or "").strip(),
        "matched": bool(verdict.matched),
        "judge_mode": verdict.mode,
        "judge_detail": verdict.detail,
        "success": success,
        "retries": int(state.get("retry_count") or 0),
        "errors": state.get("errors") or [],
        "error": error_text or state.get("error", ""),
        "elapsed": round(elapsed, 2),
        "llm_calls": counter.calls,
        "input_tokens": counter.input_tokens,
        "output_tokens": counter.output_tokens,
        "rows": 0 if result is None else int(len(result)),
        "expected": expected_preview(case, golds),
        "actual": actual_preview or (state.get("answer") or "")[:120],
        "answer": (state.get("answer") or "")[:300],
    }


def summarize(records: list) -> dict:
    """汇总指标。

    「执行成功率」只统计有标准答案的题（分母不同就该分开写）：
    该拒答/该澄清的用例本来就不该产出结果，混进去会把指标拉低，
    它们单独由「拒答/澄清正确率」衡量。
    """
    n = len(records)
    matched = sum(1 for r in records if r["matched"])
    answerable = [r for r in records if r["mode"] not in ABSTAIN_MODES]
    abstain = [r for r in records if r["mode"] in ABSTAIN_MODES]
    success = sum(1 for r in answerable if r["success"])
    retried = [r for r in records if r["retries"] > 0]
    recovered = sum(1 for r in retried if r["matched"])
    first_pass = sum(1 for r in records if r["matched"] and r["retries"] == 0)
    times = [r["elapsed"] for r in records]

    by_category = OrderedDict()
    for r in records:
        item = by_category.setdefault(r["category"], {"total": 0, "matched": 0})
        item["total"] += 1
        item["matched"] += int(r["matched"])

    routes = Counter(r["route"] or "未执行" for r in records)

    return {
        "total": n,
        "matched": matched,
        "accuracy": round(matched / n, 4) if n else 0.0,
        "answerable": len(answerable),
        "success": success,
        "success_rate": round(success / len(answerable), 4) if answerable else 0.0,
        "abstain_total": len(abstain),
        "abstain_matched": sum(1 for r in abstain if r["matched"]),
        "abstain_rate": round(sum(1 for r in abstain if r["matched"]) / len(abstain), 4)
        if abstain else 0.0,
        "first_pass": first_pass,
        "first_pass_rate": round(first_pass / n, 4) if n else 0.0,
        "retried": len(retried),
        "recovered": recovered,
        "recovery_rate": round(recovered / len(retried), 4) if retried else 0.0,
        "avg_elapsed": round(sum(times) / n, 2) if n else 0.0,
        "p50_elapsed": round(percentile(times, 0.5), 2),
        "p95_elapsed": round(percentile(times, 0.95), 2),
        "total_elapsed": round(sum(times), 1),
        "llm_calls": sum(r["llm_calls"] for r in records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "by_category": by_category,
        "routes": dict(routes),
    }


def build_report(records: list, stats: dict, args) -> str:
    """生成 markdown 报告。"""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 评测报告 {stamp}",
        "",
        f"- 模型：`{LLM_MODEL}`（temperature=0）",
        f"- 引擎：LangGraph 状态机，最大重试 {args.max_retries} 次",
        f"- 数据：`data/sales.db`（3,877 笔订单 / 20 客户 / 20 商品）",
        f"- 用例：{stats['total']} 条；洞察：{'开' if args.with_insight else '关'}；"
        f"图表：{'开' if args.with_chart else '关'}",
        f"- 总耗时：{stats['total_elapsed']} 秒",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| **准确率** | **{stats['matched']}/{stats['total']}（{stats['accuracy']:.1%}）** | 判分通过 / 全部用例 |",
        f"| 执行成功率 | {stats['success']}/{stats['answerable']}"
        f"（{stats['success_rate']:.1%}） | 拿到非空结果（仅统计有标准答案的题） |",
        f"| 拒答/澄清正确率 | {stats['abstain_matched']}/{stats['abstain_total']}"
        f"（{stats['abstain_rate']:.1%}） | 该说不的时候有没有编数 |",
        f"| 一次通过率 | {stats['first_pass']}/{stats['total']}（{stats['first_pass_rate']:.1%}） | 首次生成即正确，无需重试 |",
        f"| 自愈修复率 | {stats['recovered']}/{stats['retried']}"
        f"（{stats['recovery_rate']:.1%}） | 发生过重试并最终正确 |",
        f"| 平均耗时 | {stats['avg_elapsed']} 秒 | P50 {stats['p50_elapsed']} 秒 / P95 {stats['p95_elapsed']} 秒 |",
        f"| 模型调用 | {stats['llm_calls']} 次 | 输入 {stats['input_tokens']:,} / 输出 {stats['output_tokens']:,} tokens |",
        "",
        "## 分类准确率",
        "",
        "| 分类 | 通过 / 总数 | 准确率 |",
        "| --- | --- | --- |",
    ]
    for name, item in stats["by_category"].items():
        rate = item["matched"] / item["total"] if item["total"] else 0
        lines.append(f"| {name} | {item['matched']}/{item['total']} | {rate:.1%} |")

    lines += ["", f"路由分布：{'、'.join(f'{k} {v} 条' for k, v in stats['routes'].items())}", ""]

    failed = [r for r in records if not r["matched"]]
    lines += ["## 失败用例", ""]
    if not failed:
        lines += ["无。", ""]
    else:
        for r in failed:
            lines += [
                f"### {r['id']}（{r['category']}）",
                "",
                f"- 问题：{r['question']}",
                f"- 判定：{r['judge_detail']}",
                f"- 期望：`{r['expected']}`",
                f"- 实际：`{r['actual'] or '（无结果）'}`",
                f"- 路由：{r['route']}；重试 {r['retries']} 次；耗时 {r['elapsed']} 秒",
            ]
            if r["errors"]:
                lines.append(f"- 重试错误：{'；'.join(r['errors'])}")
            if r["statement"]:
                lines += ["", "```sql" if r["route"] == "sql" else "```python",
                          r["statement"], "```"]
            lines.append("")

    lines += ["## 评测说明（结果怎么读）", "",
              "- 判分只比结果值，不比列名与列顺序；数值走容差（默认 0.011），"
              "占比类同时接受百分比与小数两种写法。",
              "- 标准答案由金标准 SQL 在同一份库上现算，不是手工抄写，避免抄错。",
              "- **该拒答 / 该澄清两类用例没有标准答案**：判的是「该不该给数」。"
              "只要给出了带值的结果就判失败——用户拿到一个口径被替换过的数字，"
              "比被告知「查不到」严重得多。",
              "- 执行成功率的分母是「有标准答案的题」（拒答/澄清本来就不该有结果，"
              "混进去会把指标拉低），它们单独看「拒答/澄清正确率」。",
              "- 评测数据为模拟销售库；换真实业务库时只需改 `evals/cases.yaml` 里的 `gold_sql` 表名。",
              "- 成本口径：DeepSeek 输入/输出 token 数，费用按官方价自行折算。"]
    if stats["retried"] == 0:
        lines += ["- 本次没有触发重试（全部一次成功），自愈链路未在评测中体现；"
                  "其行为由 `tests/test_graph.py` 的离线用例覆盖（SQL 失败切 pandas、错误回喂、重试上限）。"]
    if stats["accuracy"] >= 0.95:
        lines += ["- 准确率接近满分说明这批题目偏基础，**不代表开放域提问都能答对**；"
                  "更难的题应持续补进评测集，而不是据此认为模型不会出错。"]

    lines += ["", "## 全部用例", "",
              "| # | ID | 分类 | 问题 | 结果 | 耗时 | 重试 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(records, 1):
        mark = "✅" if r["matched"] else "❌"
        question = r["question"].replace("|", "\\|")
        lines.append(f"| {i} | {r['id']} | {r['category']} | {question} | {mark} | "
                     f"{r['elapsed']}s | {r['retries']} |")
    lines.append("")
    return "\n".join(lines)


def save_report(records: list, stats: dict, args, report_text: str) -> tuple:
    """落盘 markdown + json。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    md_path = REPORT_DIR / f"eval-{stamp}{tag}.md"
    json_path = REPORT_DIR / f"eval-{stamp}{tag}.json"
    md_path.write_text(report_text, encoding="utf-8")
    json_path.write_text(json.dumps({
        "timestamp": stamp,
        "model": LLM_MODEL,
        "with_insight": args.with_insight,
        "with_chart": args.with_chart,
        "max_retries": args.max_retries,
        "stats": stats,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="跑评测集，统计准确率与自愈能力")
    parser.add_argument("--dry-run", action="store_true", help="只校验金标准 SQL，不调用模型")
    parser.add_argument("--only", default="", help="只跑指定 ID，逗号分隔，如 A01,B01")
    parser.add_argument("--category", default="", help="只跑指定分类")
    parser.add_argument("--limit", type=int, default=0, help="最多跑前 N 条")
    parser.add_argument("--with-insight", action="store_true", help="同时生成洞察文案（多一次模型调用）")
    parser.add_argument("--with-chart", action="store_true", help="同时生成图表（仅影响耗时）")
    parser.add_argument("--max-retries", type=int, default=2, help="最大重试次数，默认 2")
    parser.add_argument("--tag", default="", help="报告文件名后缀，便于区分多次运行")
    parser.add_argument("--out-dir", default="", help="报告输出目录，默认 evals/reports")
    parser.add_argument("--with-lessons", action="store_true",
                        help="启用修正回流（few-shot）；默认关闭，基线以冷启动口径比较")
    args = parser.parse_args()

    global REPORT_DIR
    if args.out_dir:
        REPORT_DIR = Path(args.out_dir).resolve()

    cases = select_cases(load_cases(), args)
    # 默认**不带**修正记录（use_lessons=False）：基线要反映冷启动表现。
    # 一旦沉淀了 lesson 就自动变好，历史数字就再也比不了 —— 想测"沉淀之后有多好"
    # 就显式加 --with-lessons，那属于另一个口径。
    ctx = AgentContext(use_lessons=args.with_lessons)
    agent = GraphAgent(ctx, with_chart=args.with_chart, with_insight=args.with_insight,
                       max_retries=args.max_retries)

    if args.dry_run:
        dry_run(cases, agent.ctx.db_path)
        return

    print(f"开始评测：{len(cases)} 条用例，模型 {LLM_MODEL}，"
          f"洞察={'开' if args.with_insight else '关'}\n")

    records = []
    for i, case in enumerate(cases, 1):
        golds = run_gold(case, agent.ctx.db_path)
        record = run_one(agent, case, golds, args.with_insight)
        records.append(record)
        mark = "✓" if record["matched"] else "✗"
        note = "" if record["matched"] else f"  ← {record['judge_detail']}"
        retry = f" 重试{record['retries']}" if record["retries"] else ""
        print(f"[{i:2d}/{len(cases)}] {record['id']} {mark} {record['elapsed']:5.1f}s "
              f"{record['route']}{retry}{note}")

    stats = summarize(records)
    report_text = build_report(records, stats, args)
    md_path, json_path = save_report(records, stats, args, report_text)

    print("\n" + "=" * 60)
    print(f"准确率      {stats['matched']}/{stats['total']}  "
          f"({stats['accuracy']:.1%})")
    print(f"执行成功率  {stats['success']}/{stats['answerable']}  "
          f"({stats['success_rate']:.1%})  仅统计有标准答案的题")
    print(f"拒答/澄清   {stats['abstain_matched']}/{stats['abstain_total']}  "
          f"({stats['abstain_rate']:.1%})  该不该给数")
    print(f"一次通过率  {stats['first_pass']}/{stats['total']}  "
          f"({stats['first_pass_rate']:.1%})")
    print(f"自愈修复率  {stats['recovered']}/{stats['retried']}  "
          f"({stats['recovery_rate']:.1%})")
    print(f"耗时        平均 {stats['avg_elapsed']}s，P95 {stats['p95_elapsed']}s，"
          f"合计 {stats['total_elapsed']}s")
    print(f"模型调用    {stats['llm_calls']} 次，输入 {stats['input_tokens']:,} / "
          f"输出 {stats['output_tokens']:,} tokens")
    for name, item in stats["by_category"].items():
        rate = item["matched"] / item["total"] if item["total"] else 0
        print(f"  · {name:<8} {item['matched']}/{item['total']}  {rate:.1%}")
    print("=" * 60)
    print(f"\n报告：{md_path}\n明细：{json_path}")


if __name__ == "__main__":
    main()
