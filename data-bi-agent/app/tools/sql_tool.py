"""SQL 执行工具（对应功能 F2.4，阶段 2 实现）。

设计要点：
- 只读连接：`file:...?mode=ro` 打开，从连接层就禁止写操作（双保险，不只靠关键字黑名单）；
- 校验前置：借鉴 OpenChatBI 的"先校验再执行"，只放行单条 SELECT / WITH 语句；
- 结果截断：用 fetchmany 限制行数，避免大结果撑爆内存，同时不改写原 SQL 语义。

配合 F2.3（Text-to-SQL 生成）使用，生成层与执行层解耦，便于单独测试。
"""
import re
import sqlite3
import time

import pandas as pd

from app.config import SQL_TIMEOUT_SECONDS
from app.schema.loader import get_db_path

# 修改类或危险类关键字：命中即拒绝（SQLite 的 WITH ... DELETE 也算）
FORBIDDEN_SQL_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "truncate", "attach", "detach", "pragma", "vacuum", "reindex",
    "load_extension", "writefile", "readfile",
]

MAX_ROWS_DEFAULT = 200


class UnsafeSQLError(Exception):
    """SQL 未通过安全检查。"""


def _strip_comments(sql: str) -> str:
    """去掉 -- 行注释与 /* */ 块注释，防止用注释夹带关键字绕过检查。"""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql.strip()


def check_sql_safety(sql: str) -> str:
    """校验 SQL 安全性，返回可执行的语句（已去注释、去末尾分号）。"""
    cleaned = _strip_comments(sql)
    if not cleaned:
        raise UnsafeSQLError("SQL 为空")

    body = cleaned.rstrip(";").strip()
    if ";" in body:
        raise UnsafeSQLError("只允许执行单条语句，检测到多条 SQL")

    head = re.match(r"\s*([a-zA-Z]+)", body)
    if not head or head.group(1).lower() not in ("select", "with"):
        raise UnsafeSQLError("只允许 SELECT 查询，拒绝执行其他语句")

    lowered = " " + body.lower() + " "
    for kw in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            raise UnsafeSQLError(f"SQL 包含被禁止的操作：{kw}")

    return body


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """以只读模式打开 SQLite（写操作在连接层即被拒绝）。"""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def run_sql(sql: str, db_path: str = None, max_rows: int = MAX_ROWS_DEFAULT,
            timeout: float = None) -> dict:
    """执行 SQL 并返回 DataFrame。

    返回：{ok, df, sql, rows, truncated, error, stage}

    timeout：查询超时（秒）。SQLite 没有原生的语句超时，这里用 progress handler
    按"每执行 2000 条虚拟机指令回调一次"的方式检查是否超时，超时即中断查询——
    挡住误写的笛卡尔积这类把 CPU 打满的 SQL。
    """
    db_path = db_path or get_db_path()
    timeout = SQL_TIMEOUT_SECONDS if timeout is None else timeout

    try:
        safe_sql = check_sql_safety(sql)
    except UnsafeSQLError as e:
        return {"ok": False, "stage": "safety", "error": str(e), "sql": sql, "df": None}

    deadline = time.monotonic() + timeout

    def _guard() -> int:
        return 1 if time.monotonic() > deadline else 0

    try:
        con = connect_readonly(db_path)
        try:
            con.set_progress_handler(_guard, 2000)
            cur = con.execute(safe_sql)
            fetched = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
        finally:
            con.close()
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return {"ok": False, "stage": "timeout",
                    "error": f"查询超过 {timeout:g} 秒已中止（SQL 可能写成了大笛卡尔积）",
                    "sql": safe_sql, "df": None}
        return {"ok": False, "stage": "runtime", "error": f"{type(e).__name__}: {e}",
                "sql": safe_sql, "df": None}
    except Exception as e:
        return {"ok": False, "stage": "runtime", "error": f"{type(e).__name__}: {e}",
                "sql": safe_sql, "df": None}

    truncated = len(fetched) > max_rows
    rows = fetched[:max_rows]
    df = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame()

    return {"ok": True, "df": df, "sql": safe_sql, "rows": len(df),
            "truncated": truncated, "error": None}


def list_table_columns(db_path: str = None) -> dict:
    """返回 {表名: [列名]}，供需要字段级上下文时使用。"""
    db_path = db_path or get_db_path()
    con = connect_readonly(db_path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'").fetchall()]
        return {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
                for t in tables}
    finally:
        con.close()
