"""SQL 复用：同一个问题复跑，不再让模型重写一遍 SQL（P1）。

为什么值得做
------------
提示词去重（0.14.0）之后，单条问答仍有 1,700 左右输入 token，其中绝大部分
每次回答都一样（schema + 术语 + 示例）。而"同一个问题问第二遍"在真实使用里
非常常见：看板刷新、反复核对同一个数、脚本定时跑同一批问题。
这类重复完全没必要再让模型猜一次 —— 参数是一致的，确定性回答就该被复用。

四条硬约束（每条都对应一种"复用会答错"的场景）
----------------------------------------------
1. **键里带库结构指纹**。换一个库，表名字段名全变，旧 SQL 必然跑不通。
   直接复用 `lesson_store.db_fingerprint`（只看结构不看数据），两个模块口径一致。
2. **相对时间必须绑参考日**。"上个月"昨天指 10 月，今天指 11 月，
   SQL 里已经被换算成显式日期区间 —— 不绑 today 就会拿上个月的数冒充这个月。
   `has_relative_time` 与真正注入锚点用的是**同一份规则表**，加规则自动生效。
3. **答案复用要过结果指纹**。SQL 复用是安全的（问题→查询是确定性映射），
   但"洞察文案"依赖数据本身。所以缓存里存一份结果指纹：同样的 SQL 跑出来的
   结果一字不差，才敢复用上次的结论；数据变了就复用 SQL、重新生成结论。
4. **只复用过成功的**。空结果、拒答、澄清都不进缓存 —— 把它们固化下来，
   等于把"当时答不了"变成"永远答不了"，而数据是会变的。

不复用的场景（都是会让答案变错的那种）
--------------------------------------
- 多轮追问：同一句话在上下文里指向的可能是不同的东西，改写后才能判断；
- 归因问题：要的是"两期 × 维度"的交叉表，入口参数（对比期）不在问题原文里；
- 问题里的取值库里没有：该走澄清，而不是拿一条旧 SQL 硬算；
- 时间完全超出覆盖：该说明"没有数据"，而不是给个别的区间。

另外，`teach` / `lessons --rm` 会主动清空当前库的缓存（见 `lesson_store`）：
用户既然修正了这个问题的口径，旧 SQL 就不该再被复用。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from app.sessions import home_dir
from app.tools.lesson_store import db_fingerprint
from app.tools.sql_tool import UnsafeSQLError, check_sql_safety
from app.tools.time_anchor import has_relative_time, out_of_range_answer

MAX_ENTRIES = 200
"""单库最多存多少条。超出按"最久没用"淘汰 —— 缓存是加速手段，不能无限长。"""

MAX_SQL_CHARS = 2000
MAX_INSIGHT_CHARS = 1200
"""洞察文案的存放上限。超长的多半是模型跑题了，存进去也只是浪费磁盘。"""


class CacheError(Exception):
    """缓存条目不可用（SQL 不安全、格式不对）。调用方应视为"没命中"。"""


_PUNCT = re.compile(r"[\s，。？！、；：\"'（）()【】\[\]《》<>·…—\-_/\\,.?!;:|]+")


def _context():
    """默认上下文（惰性导入，避开与 context.py 的循环依赖）。"""
    from app.context import AgentContext
    return AgentContext()


# ---------------------------------------------------------------- 位置
def cache_dir() -> Path:
    return home_dir() / "sqlcache"


def cache_file(ctx=None) -> Path:
    """按库指纹分文件：不同库的问题互不干扰，删一个库的缓存不影响别的。"""
    return cache_dir() / f"{db_fingerprint(ctx)}.json"


# ---------------------------------------------------------------- 键
def normalize_question(question: str) -> str:
    """归一化：去标点空白、统一大小写。

    「各区域的销售额是多少？」与「各区域的销售额是多少」是同一个问题，
    问号多了个全角的区别不该让我们再花一次钱。
    """
    return _PUNCT.sub("", (question or "").strip()).lower()


def _catalog_stamp(ctx) -> str:
    """术语/口径文本的指纹：词典改了，旧 SQL 的口径未必还对。"""
    try:
        terms = ctx.terms or ""
    except Exception:
        return "no-terms"
    return hashlib.sha1(terms.encode("utf-8")).hexdigest()[:8]


def _asof(question: str, ctx) -> str:
    """把「相对时间的换算结果」绑进键里 —— 注意不是绑"今天是哪天"。

    为什么绑区间而不是绑 today：「上个月」在 1 月 15 日和 1 月 20 日都指向
    2025-12，绑 today 会让它每周白白失效一次；而 2 月 1 日指向 2026-01，
    必须与前者分开。绑定换算出来的日期区间两头都对：同一区间可复用，
    跨到另一区间立刻失效。

    判据用 `has_relative_time`，它和 `resolve_time_anchor` 共用 `_PATTERNS`
    规则表 —— 以后新增"近两周"这类说法两处同时生效，不会出现
    "锚点注入了但缓存没绑日期"的漏（那种错没有任何报错）。
    """
    try:
        if not has_relative_time(question):
            return ""
        anchors = ctx.time_anchors(question)
        if not anchors:
            # 认得出是相对时间、却换算不出区间（天数超上限等）→ 保守绑 today
            return ctx.today.isoformat()
        return "|".join(f"{a.start.isoformat()}~{a.end.isoformat()}" for a in anchors)
    except Exception:
        return "unknown-today"


def cache_key(question: str, ctx=None) -> str:
    """一次复用的身份：问题 + 库结构 + 词典口径 + 参考日。"""
    ctx = ctx or _context()
    parts = [
        normalize_question(question),
        db_fingerprint(ctx),
        _catalog_stamp(ctx),
        _asof(question, ctx),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 结果指纹
def result_fingerprint(df) -> str:
    """结果表的指纹：列名 + 逐行哈希。

    用途只有一个 —— 判断"同样的 SQL 是不是跑出了同样的数据"，
    相同才复用上次的洞察文案。算不出来（含不可哈希的嵌套对象）返回空串，
    调用方据此跳过文案复用（宁可多调一次模型，不给过期结论）。
    """
    if df is None:
        return ""
    try:
        if len(df) == 0:
            return "empty"
        hashed = pd.util.hash_pandas_object(df, index=True).values
        digest = hashlib.sha1()
        digest.update(",".join(str(c) for c in df.columns).encode("utf-8"))
        digest.update(hashed.tobytes())
        return digest.hexdigest()[:16]
    except Exception:
        return ""


# ---------------------------------------------------------------- 读
def _load_all(ctx=None) -> list:
    file = cache_file(ctx)
    if not file.exists():
        return []
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []          # 坏文件当空处理：缓存坏了最多是"没加速"，不能影响问答
    return [item for item in (data or []) if isinstance(item, dict) and item.get("key")]


def _save_all(entries: list, ctx=None) -> None:
    """原子写：先写临时文件再 replace，避免写到一半断电留下半截 JSON。"""
    file = cache_file(ctx)
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)
        os.replace(tmp, file)
    except OSError:
        pass               # 写不进去（只读部署、磁盘满）就不缓存，主流程照常


def _skip_reason(question: str, ctx, has_history: bool = False) -> str:
    """不能复用的原因；返回空串表示可以查缓存。"""
    if not (question or "").strip():
        return "问题为空"
    if not getattr(ctx, "use_cache", True):
        return "缓存已关闭"
    if not ctx.has_db:
        return "单表模式"
    if has_history:
        return "多轮追问"

    # 取值库里没有 → 该走澄清。缓存一条旧 SQL 会把这个"该澄清"盖掉。
    try:
        from app.tools.enum_guard import find_unknown_enum
        if find_unknown_enum(question, ctx.enums):
            return "取值待澄清"
    except Exception:
        return "取值校验不可用"

    # 归因要比两期，入口参数不在问题原文里，SQL 也不止一条普通语句。
    try:
        from app.tools.attribution import detect_attribution
        if detect_attribution(question, ctx.enums):
            return "归因问题"
    except Exception:
        return "归因识别不可用"

    # 时间完全超出覆盖范围 → 该直接说明"没有数据"，不能拿旧 SQL 算个别的区间。
    try:
        anchors = ctx.time_anchors(question)
        if out_of_range_answer(anchors, ctx.date_coverage):
            return "时间超出覆盖"
    except Exception:
        return "时间校验不可用"
    return ""


def find_cached_sql(question: str, ctx=None,
                    has_history: bool = False) -> Tuple[Optional[dict], str]:
    """查缓存。返回 `(条目, 原因)`：命中时原因里是人话说明，未命中时是短原因。

    命中后会顺手把 `hits` / `last_used` 写回 —— 淘汰策略要用它，
    而且"这条被复用过几次"本身就是判断"值不值得留着"的依据。
    """
    ctx = ctx or _context()
    reason = _skip_reason(question, ctx, has_history=has_history)
    if reason:
        return None, reason

    key = cache_key(question, ctx)
    entries = _load_all(ctx)
    for item in entries:
        if item.get("key") != key:
            continue
        sql = str(item.get("sql") or "").strip()
        if not sql:
            return None, "条目损坏"
        try:
            # 缓存文件在本地磁盘上，理论上可以被手改。执行前再过一次安全检查：
            # 一条被塞进来的 DROP 若照跑，复用就成了投毒通道。
            check_sql_safety(sql)
        except UnsafeSQLError:
            return None, "条目不安全"

        item["hits"] = int(item.get("hits") or 0) + 1
        item["last_used"] = datetime.now().isoformat(timespec="seconds")
        _save_all(entries, ctx)
        return item, f"复用上次的查询（第 {item['hits']} 次）"
    return None, "未命中"


# ---------------------------------------------------------------- 写
def remember_sql(question: str, sql: str, ctx=None, df=None,
                 insight: str = "") -> Optional[dict]:
    """把一条跑通的「问题 → SQL」存进缓存（同问题覆盖，不堆积）。

    只该在**真的给出了答案**之后调用：空结果、拒答、澄清都不进来。
    """
    ctx = ctx or _context()
    question = (question or "").strip()
    text = (sql or "").strip()
    if not question or not text:
        return None
    if not getattr(ctx, "use_cache", True) or not ctx.has_db:
        return None
    if len(text) > MAX_SQL_CHARS:
        return None        # 超长语句多半是模型跑偏了，存了也不敢复用
    # 不能复用的场景就别存了（归因、取值待澄清、时间超范围…）：
    # 判据和 lookup 共用一份，否则会出现"存了一堆永远查不到的条目"。
    if _skip_reason(question, ctx):
        return None
    try:
        text = check_sql_safety(text)
    except UnsafeSQLError:
        return None

    note = (insight or "").strip()
    if len(note) > MAX_INSIGHT_CHARS:
        note = ""          # 超长结论不存：宁可不复用文案，也不截断成半句话

    key = cache_key(question, ctx)
    entries = _load_all(ctx)
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "key": key,
        "question": question[:300],
        "sql": text,
        "result_fp": result_fingerprint(df),
        "insight": note,
        "updated": now,
        # 写入即视为"刚用过"：淘汰按这个时间排，同一秒内并列时靠稳定排序，
        # 行为仍然可预期（先写的先淘汰）。
        "last_used": now,
    }

    for item in entries:
        if item.get("key") == key:
            # 保留首次创建时间与累计命中数 —— 它们是"这条缓存的实际价值"
            payload["created"] = item.get("created") or now
            payload["hits"] = int(item.get("hits") or 0)
            item.clear()
            item.update(payload)
            break
    else:
        payload["created"] = now
        payload["hits"] = 0
        entries.append(payload)

    if len(entries) > MAX_ENTRIES:
        entries.sort(key=lambda e: str(e.get("last_used") or e.get("created") or ""))
        entries = entries[-MAX_ENTRIES:]
    _save_all(entries, ctx)
    return payload


def forget_sql(question: str, ctx=None) -> bool:
    """删掉一条缓存（命中后执行失败、SQL 已过时时用）。"""
    ctx = ctx or _context()
    key = cache_key(question, ctx)
    entries = _load_all(ctx)
    kept = [item for item in entries if item.get("key") != key]
    if len(kept) == len(entries):
        return False
    _save_all(kept, ctx)
    return True


def invalidate_cache(ctx=None) -> int:
    """清空当前库的全部缓存，返回删掉的条数（`teach` 之后调用）。"""
    ctx = ctx or _context()
    entries = _load_all(ctx)
    if not entries:
        return 0
    try:
        cache_file(ctx).unlink()
    except OSError:
        _save_all([], ctx)
    return len(entries)


def cache_stats(ctx=None) -> dict:
    """缓存概况（给 `logs` / 调试用）：条数、累计复用次数、最近使用时间。"""
    ctx = ctx or _context()
    entries = _load_all(ctx)
    return {
        "entries": len(entries),
        "hits": sum(int(e.get("hits") or 0) for e in entries),
        "last_used": max((str(e.get("last_used") or "") for e in entries), default=""),
        "path": str(cache_file(ctx)),
    }
