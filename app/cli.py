"""命令行问答入口（对应步骤 S1.7 / S2.7，功能 F4.1 的命令行版）。

用法：
  交互模式：python -m app.cli
  单次提问：python -m app.cli "各区域的销售额是多少？"

可选参数：
  --no-chart     不生成图表
  --no-insight   不生成洞察文案（省一次模型调用）
  --pandas       强制走单表 pandas 模式（默认优先 Text-to-SQL）
  --quiet        不打印生成的 SQL / 代码
"""
import sys

from app.agent import DataAgent
from app.nodes.execute import format_result

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文打印报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def render(agent: DataAgent, question: str, flags: set) -> None:
    """回答单个问题并打印全过程。"""
    print(f"\n===== 问题：{question} =====")

    res = agent.ask(
        question,
        with_chart="--no-chart" not in flags,
        with_insight="--no-insight" not in flags,
        force_pandas="--pandas" in flags,
    )

    if not res.ok:
        print("\n[未能回答]")
        for err in res.errors:
            print(f"  · {err}")
        return

    route_label = "SQLite 多表（Text-to-SQL）" if res.route == "sql" else "单表宽表（Text-to-Pandas）"
    print(f"\n[路由] {route_label}")

    if "--quiet" not in flags:
        print(f"\n[生成的{'SQL' if res.route == 'sql' else '代码'}]")
        print(res.statement)

    print(f"\n[分析结果]（{len(res.df)} 行）")
    print(format_result(res.df))

    if res.chart_path:
        print(f"\n[图表] {res.chart_label} → {res.chart_path}")
    elif res.chart_type == "none" and "--no-chart" not in flags:
        print("\n[图表] 结果不适合可视化，已跳过")

    if res.insight:
        print(f"\n[洞察] {res.insight}")

    # 路由回退时说明原因，便于排查
    for err in res.errors:
        print(f"\n[降级说明] {err}")


def main() -> None:
    argv = sys.argv[1:]
    flags = {a for a in argv if a.startswith("--")}
    question = " ".join(a for a in argv if not a.startswith("--")).strip()

    agent = DataAgent()

    if question:
        render(agent, question, flags)
        return

    print("数据分析 Agent 已就绪（输入问题，输入 q 退出）")
    print(f"当前数据模式：{agent.mode_label}")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if q.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if not q:
            continue
        try:
            render(agent, q, flags)
        except Exception as e:  # 兜底，避免单次失败中断会话
            print(f"[错误] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
