"""命令行问答入口（对应步骤 S1.7 / 功能 F4.1 的命令行版）。

用法：
  交互模式：python -m app.cli
  单次提问：python -m app.cli "各区域的销售额是多少？"
"""
import sys

import pandas as pd

from app.nodes.execute import execute_code, format_result
from app.nodes.generate import generate_code
from app.schema.loader import build_schema_text, load_catalog_terms, load_dataframe

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文打印报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def answer(df: pd.DataFrame, schema_info: str, terms: str, question: str) -> None:
    """回答单个问题：生成代码 → 执行 → 展示结果。"""
    print(f"\n===== 问题：{question} =====")
    code = generate_code(schema_info, terms, question)
    print(f"\n[生成的代码]\n{code}")

    res = execute_code(code, df)
    if not res["ok"]:
        print(f"\n[执行失败 · {res.get('stage')}] {res['error']}")
        return
    print(f"\n[分析结果]\n{format_result(res['result'])}")


def main() -> None:
    df = load_dataframe()
    schema_info = build_schema_text(df)
    terms = load_catalog_terms()

    # 单次提问模式
    if len(sys.argv) > 1:
        answer(df, schema_info, terms, " ".join(sys.argv[1:]))
        return

    # 交互模式
    print("数据分析 Agent 已就绪（输入问题，输入 q 退出）")
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
            answer(df, schema_info, terms, q)
        except Exception as e:  # 兜底，避免单次失败中断会话
            print(f"[错误] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
