"""代码执行节点（对应功能 F2.2，阶段 1 实现；F5.2 / S5.3 阶段 5 加固）。

借鉴点：
- OpenChatBI：生成后"先校验、再执行"，把不安全/明显错误的代码挡在执行之前；
- PandasAI：代码执行放在受限环境（其 CVE-2024-12366 即提示注入导致任意代码执行，
  说明执行层必须设防）。

三层防护（阶段 5 加固后）：
1. **静态校验**：用 AST 而不是纯字符串匹配——禁止 import、禁止 dunder 属性
   （`__class__.__bases__` 这类逃逸手法）、禁止写盘方法，字符串黑名单仍保留作兜底；
2. **受限命名空间**：只给白名单内置函数，`df` 传副本；
3. **执行超时**：代码跑在守护线程里，超过时限就放弃等待并返回超时错误，
   避免死循环把会话卡死。

超时的已知局限（如实说明）：Python 无法强杀线程，若生成代码是**不释放 GIL 的
C 层紧密计算**（如 `sum(range(10**8))`），主线程要等它跑完才能恢复；
死循环、pandas/numpy 长任务这类真正会"卡死会话"的场景不受影响。
需要强隔离时用 Docker（见 README 部署章节）。

此外对结果行数设上限，防止把超大结果塞进内存与前端。
"""
import ast
import threading
import traceback

import pandas as pd

from app.config import EXEC_TIMEOUT_SECONDS, MAX_CODE_NODES, MAX_RESULT_ROWS

# 字符串黑名单：AST 已禁掉 import 与危险名，这里防的是拼接、注释等边角绕过
FORBIDDEN_PATTERNS = [
    "import os", "import sys", "import subprocess", "import socket",
    "import shutil", "__import__", "eval(", "exec(", "compile(",
    "open(", "input(", "breakpoint(", "globals()", "locals()",
    "os.system", "os.remove", "os.rmdir", "shutil.rmtree",
    "requests.", "urllib", "pickle", "pathlib",
]

# 禁止出现的名字（即使绕过内置白名单也不给用）
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "breakpoint", "__import__",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "memoryview", "exit", "quit", "help", "__builtins__",
}

# 禁止访问的属性/方法：dunder 逃逸链与写盘、进程操作
FORBIDDEN_ATTRS = {
    "to_csv", "to_excel", "to_sql", "to_pickle", "to_parquet", "to_feather",
    "to_hdf", "to_stata", "to_clipboard",
    "system", "popen", "spawn", "remove", "rmdir", "unlink", "chmod", "kill",
}

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


def check_safety(code: str, max_nodes: int = None) -> None:
    """执行前的安全检查（借鉴 OpenChatBI 的"校验在前"）。

    用 AST 遍历代替纯文本匹配，能挡住 `import  os`（多空格）、
    `df.__class__.__bases__` 这类字符串匹配看不出来的写法。
    """
    max_nodes = max_nodes or MAX_CODE_NODES

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"代码语法错误：{e}") from e

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > max_nodes:
        raise UnsafeCodeError(f"代码过于复杂（{node_count} 个语法节点，上限 {max_nodes}）")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("代码包含被禁止的操作：import 导入模块")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"代码包含被禁止的操作：{node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCodeError(f"代码尝试访问内部属性：{node.attr}")
            if node.attr in FORBIDDEN_ATTRS:
                raise UnsafeCodeError(f"代码包含被禁止的操作：{node.attr}")

    # 字符串兜底：防止用字符串拼接、注释等方式夹带关键字
    lowered = code.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in lowered:
            raise UnsafeCodeError(f"代码包含被禁止的操作：{pattern}")


def _truncate(result, max_rows: int):
    """结果行数截断；返回 (结果, 是否被截断)。"""
    if isinstance(result, pd.DataFrame) and len(result) > max_rows:
        return result.head(max_rows), True
    if isinstance(result, pd.Series) and len(result) > max_rows:
        return result.head(max_rows), True
    return result, False


def execute_code(code: str, df: pd.DataFrame, timeout: float = None,
                 max_rows: int = None) -> dict:
    """在受限命名空间执行代码，返回 {ok, result, error, stage, truncated}。

    约定：生成的代码需把答案赋值给变量 result。
    """
    timeout = EXEC_TIMEOUT_SECONDS if timeout is None else timeout
    max_rows = MAX_RESULT_ROWS if max_rows is None else max_rows

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
    box: dict = {}

    def _work() -> None:
        try:
            exec(code, namespace)  # noqa: S102 —— 已做 AST 校验 + 受限命名空间
            box["done"] = True
        except BaseException as e:      # noqa: BLE001 —— 生成代码什么都可能抛
            box["error"] = f"{type(e).__name__}: {e}"
            box["traceback"] = traceback.format_exc(limit=3)

    worker = threading.Thread(target=_work, name="agent-exec", daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        # 线程杀不掉，但可以不再等它：主流程立刻返回，界面不会卡死
        return {
            "ok": False, "result": None, "stage": "timeout",
            "error": f"代码执行超过 {timeout:g} 秒已中止（疑似死循环或计算量过大）",
        }

    if "error" in box:
        return {
            "ok": False, "result": None,
            "error": box["error"],
            "traceback": box.get("traceback", ""),
            "stage": "runtime",
        }

    if "result" not in namespace:
        return {"ok": False, "result": None,
                "error": "代码未生成 result 变量", "stage": "result"}

    result, truncated = _truncate(namespace["result"], max_rows)
    out = {"ok": True, "result": result, "error": None, "truncated": truncated}
    if truncated:
        out["note"] = f"结果超过 {max_rows} 行，已截断"
    return out


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
