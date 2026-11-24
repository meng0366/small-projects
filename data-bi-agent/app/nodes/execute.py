"""代码执行节点（对应功能 F2.2，阶段 1 实现；F5.2 阶段 5 再加固）。

借鉴点：
- OpenChatBI：生成后"先校验、再执行"，把不安全/明显错误的代码挡在执行之前；
- PandasAI：代码执行放在受限环境（其 CVE-2024-12366 即提示注入导致任意代码执行，
  说明执行层必须设防）。

当前为轻量防护（语法检查 + 危险调用黑名单 + 受限内置函数）。
更强的隔离（子进程/超时/Docker）放在阶段 5 的 F5.2 完成。
"""
import traceback

import pandas as pd

# 危险关键字：命中即拒绝执行
FORBIDDEN_PATTERNS = [
    "import os", "import sys", "import subprocess", "import socket",
    "import shutil", "__import__", "eval(", "exec(", "compile(",
    "open(", "input(", "breakpoint(", "globals()", "locals()",
    "os.system", "os.remove", "os.rmdir", "shutil.rmtree",
    "requests.", "urllib", "pickle", "pathlib",
]

# 允许使用的内置函数（其余全部屏蔽）
SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "round": round,
    "abs": abs, "sorted": sorted, "range": range, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "float": float, "bool": bool, "print": print, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
    "isinstance": isinstance,
}


class UnsafeCodeError(Exception):
    """代码未通过安全检查。"""


def check_safety(code: str) -> None:
    """执行前的安全检查（借鉴 OpenChatBI 的"校验在前"）。"""
    lowered = code.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in lowered:
            raise UnsafeCodeError(f"代码包含被禁止的操作：{pattern}")

    # 语法检查：编译一次，提前暴露语法错误
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"代码语法错误：{e}") from e


def execute_code(code: str, df: pd.DataFrame) -> dict:
    """在受限命名空间执行代码，返回 {ok, result, error}。

    约定：生成的代码需把答案赋值给变量 result。
    """
    try:
        check_safety(code)
    except UnsafeCodeError as e:
        return {"ok": False, "result": None, "error": str(e), "stage": "safety"}

    # 受限命名空间：只用白名单内置函数，屏蔽 __builtins__
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "df": df.copy(),   # 传副本，避免代码改动原始数据
    }

    try:
        exec(code, namespace)  # noqa: S102 —— 已做安全检查 + 受限命名空间
    except Exception as e:
        return {
            "ok": False,
            "result": None,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
            "stage": "runtime",
        }

    if "result" not in namespace:
        return {"ok": False, "result": None,
                "error": "代码未生成 result 变量", "stage": "result"}

    return {"ok": True, "result": namespace["result"], "error": None}


def format_result(result, max_rows: int = 50) -> str:
    """把执行结果格式化为可读文本。"""
    if isinstance(result, pd.DataFrame):
        shown = result.head(max_rows)
        text = shown.to_string(index=False)
        if len(result) > max_rows:
            text += f"\n（共 {len(result)} 行，仅显示前 {max_rows} 行）"
        return text
    if isinstance(result, pd.Series):
        return result.to_string()
    return str(result)
