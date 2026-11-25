"""结构化输出：命令行 `-f json` 与 MCP 工具共用同一份 payload。

为什么单独一个模块：
- 同一个问题，命令行要给人看表、脚本要读 JSON、MCP 客户端（另一个 Agent）
  要读 JSON —— 后者对字段的要求更硬：要有类型、要能判断失败、要能知道
  "这份数据是不是被截断了"。三处各写一份必然漂移，所以集中在这里。
- 全部纯函数，不打印、不写盘，方便离线测试。

一个刻意的取舍：**MCP 侧必须截断数据**。
命令行输出几千行是用户自己的屏幕，MCP 返回几千行进的是另一个模型的上下文窗口，
会把对方的 token 预算吃掉。截断时明确写出 `truncated / total_rows`，让对面知道
"这不是全部"，而不是悄悄给一部分就完事。
"""
import json
import sqlite3
from typing import Any, Optional

from app.render import render
from app.schema.loader import list_tables, load_catalog


def db_columns(db_path: str) -> dict:
    """读取每张表的字段名与行数（只读连接，仅用于展示与描述）。

    用 `mode=ro` 打开：这个函数只该看结构，不该有任何写入可能。
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        result = {}
        for name in list_tables(db_path):
            rows = con.execute(f'PRAGMA table_info("{name}")').fetchall()
            count = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            result[name] = {"columns": [r[1] for r in rows], "rows": count}
        return result
    finally:
        con.close()


def schema_payload(ctx) -> dict:
    """库结构 + 语义词典的可序列化视图（`schema` 命令与 MCP `get_schema` 共用）。"""
    catalog = load_catalog()
    labels = {}
    for table in catalog.get("tables") or []:
        for column, meta in (table.get("columns") or {}).items():
            if isinstance(meta, dict) and meta.get("label"):
                labels[f"{table.get('name')}.{column}"] = meta["label"]

    tables = []
    if ctx.has_db:
        for name, meta in db_columns(ctx.db_path).items():
            tables.append({
                "name": name,
                "rows": meta["rows"],
                "columns": [
                    {"name": col, "label": labels.get(f"{name}.{col}", "")}
                    for col in meta["columns"]
                ],
            })

    return {
        "mode": ctx.mode_label,
        "db": ctx.db_path,
        "tables": tables,
        "terms": catalog.get("terms") or {},
        "relationships": catalog.get("relationships") or [],
    }


def _attribution_payload(att: dict) -> Optional[dict]:
    """把归因结果整理成嵌套结构（含二级下钻）。没有可信归因时返回 None。"""
    if not att.get("ok"):
        return None
    payload = {
        "compare": {"base": att.get("base_label"), "current": att.get("current_label")},
        "dimension": att.get("dim"),
        "metric": att.get("metric"),
        "base_total": att.get("base_total"),
        "current_total": att.get("current_total"),
        "delta": att.get("delta"),
        "pct": att.get("pct"),
        "additive": att.get("additive"),
        "rows": json.loads(render(att["table"], "json")),
        "notes": att.get("notes") or [],
    }
    drill = att.get("drill")
    if drill:
        payload["drill"] = {
            "dimension": drill.get("dim"),
            "item": drill.get("item"),
            "item_delta": drill.get("item_delta"),
            "closure_ok": drill.get("closure_ok"),
            "rows": json.loads(render(drill["table"], "json")),
        }
    return payload


def build_payload(state: dict, question: str, max_rows: int = 0) -> dict:
    """一次问答的结构化结果。

    `max_rows > 0` 时只保留前 N 行，并显式标注 `truncated` / `total_rows` ——
    给另一个模型喂数据时，这个标注比数据本身还重要。
    """
    df = state.get("result")
    rows = [] if df is None else json.loads(render(df, "json"))
    total = len(rows)
    truncated = bool(max_rows and total > max_rows)
    if truncated:
        rows = rows[:max_rows]

    return {
        "question": question,
        "rewritten": state.get("rewritten_question") or "",
        "trace_id": state.get("trace_id") or "",
        "plan": {
            "used": bool(state.get("plan")),
            "reason": state.get("plan_reason") or "",
            "content": state.get("plan"),
            "issues": state.get("plan_issues") or [],
        },
        "peek": {
            "used": bool(state.get("peek_used")),
            "rows": int(state.get("peek_rows") or 0),
            "note": state.get("peek_note") or "",
            "error": state.get("peek_error") or "",
        },
        # SQL 复用（P1）：调用方要能看出"这次的 SQL 是模型现写的还是复用的"，
        # 前者可能受模型波动影响，后者是上次已经跑通过的那一条。
        "cache": {
            "hit": bool(state.get("cache_hit")),
            "note": state.get("cache_note") or "",
        },
        "ok": df is not None,
        "route": state.get("route") or "",
        "statement": state.get("statement") or "",
        "columns": [] if df is None else [str(c) for c in df.columns],
        "rows": total,
        "row_limit": max_rows,
        "truncated": truncated,
        "data": rows,
        "attribution": _attribution_payload(state.get("attribution") or {}),
        "insight": state.get("insight") or "",
        "suggestions": state.get("suggestions") or [],
        "retries": int(state.get("retry_count") or 0),
        "errors": state.get("errors") or [],
        "elapsed": state.get("_elapsed"),
    }


def dumps(payload: Any) -> str:
    """统一序列化：中文不转义、日期等对象转字符串。

    MCP 工具的返回值会被另一个模型直接阅读，`\\u4e2d\\u6587` 那种转义
    既费 token 又难读，所以固定 `ensure_ascii=False`。
    """
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
