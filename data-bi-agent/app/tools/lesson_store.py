"""修正回流：把「问错的问题 → 对的输出」沉淀成 few-shot（P1-1）。

借鉴 Datus-agent 的 correction 回流思路：人工修正过的问答不该只解决一次，
应当沉进上下文层，下次遇到相似问题直接给出正确写法。

三条硬约束（都是踩过坑才写的）：

1. **按库指纹隔离**。few-shot 里的表名、字段名强绑定 schema，换一个库还注入
   就会教模型写出不存在的字段 —— 这正是"静默映射"那类错误的放大器。
   文件名带上库指纹，天然隔离，不同库互不影响。
2. **只收能跑的只读查询**。lesson 会被拼进 Prompt，等于"可信上下文"；
   一条带 DROP 的 lesson 就是投毒入口。入库前过 `check_sql_safety`，
   并默认真跑一遍（`--force` 可跳过），存进去的东西必须是能执行的。
3. **只注入相关的**。always-on 的 few-shot 会挤占上下文并误导模型
   （问"各区域销售额"却看到一条"退货率"的写法）。按中文 bigram 相似度
   + 枚举取值重叠排序，低于阈值的一条都不给。

`kind=defer` 的 lesson 是"这类问题该拒答"的示范（output 为 `DEFER: 原因`），
它治的是"答不了却硬编一个数"——比教怎么写 SQL 更有价值。
"""
from __future__ import annotations

import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.tools.sql_tool import UnsafeSQLError, check_sql_safety, list_table_columns, run_sql

# 不在这里 import AgentContext：context.py 要反向导入本模块（cached_property lessons），
# 顶层导入会形成循环。下面统一用 `_context()` 惰性取默认上下文。

MAX_QUESTION_CHARS = 300
MAX_OUTPUT_CHARS = 900
MAX_LESSONS_IN_PROMPT = 3
MAX_INJECT_CHARS = 2200
MIN_SCORE = 0.42
"""相关度门槛：低于它宁可不注入。

实测「华南地区的销售额」对一条「华南的销售额」示范是 0.49，真同类；
而「每个月的销售额走势」对同一条只有 0.33 —— 共享"销售额"三个字的偶然重叠
集中在 0.25–0.4 这一带，门槛就卡在这之间。"""

VALUE_BONUS = 0.25
"""问题里出现同一条 lesson 的枚举取值（如都提到"华南"）时加分。"""

_PUNCT = re.compile(r"[\s，。？！、；：\"'（）()【】\[\]《》<>·…—\-_/\\,.?!;:|]+")
_DEFER_HEAD = re.compile(r"(?is)^\s*DEFER\s*[:：]")


class LessonError(Exception):
    """lesson 入库校验失败（内容为空、SQL 不安全、跑不通等）。"""


def _drop_sql_cache(ctx=None, path: Optional[str] = None) -> None:
    """用户修正了这类问题 → 该类问题的历史 SQL 立刻作废。

    不做这一步会有个很隐蔽的坑：teach 之后问题原文没变，SQL 复用直接命中
    修正**之前**的那条语句，于是"教过了却还是错的答案"，而日志里一切正常。
    只在走默认存储位置时清（测试传 path 时不该动全局缓存）。
    """
    if path:
        return
    try:
        from app.tools import sql_cache
        sql_cache.invalidate_cache(ctx)
    except Exception:
        pass          # 清缓存失败不该让 teach 失败


# ---------------------------------------------------------------- 存取位置
def _context():
    """默认上下文（惰性导入，避开与 context.py 的循环依赖）。"""
    from app.context import AgentContext
    return AgentContext()


def _home() -> Path:
    # 与会话共用 AGENT_HOME，测试与多环境隔离沿用同一套约定
    from app.sessions import home_dir
    return home_dir()


def lessons_dir() -> Path:
    return _home() / "lessons"


def db_fingerprint(ctx: Optional[AgentContext] = None) -> str:
    """库指纹：表名 + 字段名 + 字段数的短 hash。

    只取结构不取数据：数据会天天变，结构相对稳定；也避免指纹里带出业务数据。
    没有数据库（CSV 单表模式）时退化为宽表列名。
    """
    ctx = ctx or _context()
    parts: List[str] = []
    if ctx.has_db:
        try:
            for table, columns in sorted(list_table_columns(ctx.db_path).items()):
                parts.append(f"{table}({','.join(columns)})")
        except Exception:
            parts.append("db-unreadable")
    else:
        try:
            parts.append("csv(" + ",".join(str(c) for c in ctx.df.columns) + ")")
        except Exception:
            parts.append("csv-unreadable")
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:10]


def lessons_path(ctx: Optional[AgentContext] = None) -> Path:
    return lessons_dir() / f"{db_fingerprint(ctx)}.jsonl"


# ---------------------------------------------------------------- 读
def load_lessons(ctx: Optional[AgentContext] = None,
                 path: Optional[str] = None) -> List[dict]:
    """读取当前库的全部 lesson（按创建时间升序）。

    坏行直接跳过：lesson 是"锦上添花"的上下文，不能因为一行脏数据就让问答失败。
    """
    file = Path(path) if path else lessons_path(ctx)
    if not file.exists():
        return []

    items: List[dict] = []
    try:
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("question") and item.get("output"):
                    items.append(item)
    except OSError:
        return []
    return items


def _next_id(items: List[dict]) -> str:
    """L001 递增；用现有最大编号 +1，删掉中间某条也不会重号。"""
    biggest = 0
    for item in items:
        match = re.match(r"^L(\d+)$", str(item.get("id") or ""))
        if match:
            biggest = max(biggest, int(match.group(1)))
    return f"L{biggest + 1:03d}"


# ---------------------------------------------------------------- 写
def _normalize_output(output: str, kind: Optional[str]) -> str:
    """规范化输出内容并判定类型：SQL 走安全检查，DEFER 只校验格式。"""
    text = (output or "").strip()
    if not text:
        raise LessonError("正确输出不能为空")
    if len(text) > MAX_OUTPUT_CHARS:
        raise LessonError(f"正确输出太长了（{len(text)} 字符 > {MAX_OUTPUT_CHARS}），"
                          "few-shot 里只放能说明写法的最小片段")
    if kind == "defer" or _DEFER_HEAD.match(text):
        if not _DEFER_HEAD.match(text):
            raise LessonError("kind=defer 时输出必须以 `DEFER:` 开头")
        return text
    try:
        return check_sql_safety(text)
    except UnsafeSQLError as e:
        raise LessonError(f"这条 SQL 不能作为 few-shot：{e}") from e


def add_lesson(question: str, output: str, note: str = "", kind: Optional[str] = None,
               ctx: Optional[AgentContext] = None, path: Optional[str] = None,
               verify: bool = True) -> dict:
    """新增/覆盖一条 lesson。

    同一个问题（归一化后相同）重复 teach 视为**修正之前的记录**，覆盖而不是堆积 ——
    否则两条互相矛盾的示范会同时进 Prompt，模型只能随机挑一个。
    """
    question = (question or "").strip()
    if not question:
        raise LessonError("问题不能为空")
    if len(question) > MAX_QUESTION_CHARS:
        raise LessonError(f"问题太长了（{len(question)} 字符 > {MAX_QUESTION_CHARS}）")

    text = _normalize_output(output, kind)
    is_defer = bool(_DEFER_HEAD.match(text))

    if verify and not is_defer:
        ctx = ctx or _context()
        if not ctx.has_db:
            raise LessonError("当前是单表 CSV 模式，无法验证 SQL；加 --force 可跳过校验")
        result = run_sql(text, ctx.db_path)
        if not result["ok"]:
            raise LessonError(
                f"这条 SQL 在当前库上执行失败（{result['stage']} 阶段）：{result['error']}\n"
                "存进去会教模型写错语句，已拒绝。确认要存可加 --force")

    file = Path(path) if path else lessons_path(ctx)
    items = load_lessons(ctx, str(file))

    key = _question_key(question)
    for existing in items:
        if _question_key(existing.get("question", "")) == key:
            existing.update({"question": question, "output": text,
                             "kind": "defer" if is_defer else "sql",
                             "note": note or existing.get("note", ""),
                             "updated": datetime.now().isoformat(timespec="seconds")})
            _write(file, items)
            _drop_sql_cache(ctx, path)
            return existing

    lesson = {
        "id": _next_id(items),
        "question": question,
        "output": text,
        "kind": "defer" if is_defer else "sql",
        "note": (note or "").strip(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "teach",
    }
    items.append(lesson)
    _write(file, items)
    _drop_sql_cache(ctx, path)
    return lesson


def remove_lesson(lesson_id: str, ctx: Optional[AgentContext] = None,
                  path: Optional[str] = None) -> bool:
    """按 id 删除；返回是否真的删掉了（id 不存在返回 False，由调用方给提示）。"""
    file = Path(path) if path else lessons_path(ctx)
    items = load_lessons(ctx, str(file))
    kept = [item for item in items if str(item.get("id")) != str(lesson_id)]
    if len(kept) == len(items):
        return False
    _write(file, kept)
    _drop_sql_cache(ctx, path)
    return True


def _write(file: Path, items: List[dict]) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 挑选
def _question_key(question: str) -> str:
    """归一化：去掉标点与空白，用来判"是不是同一个问题"。"""
    return _PUNCT.sub("", (question or "").strip())


def _bigrams(text: str, strip=()) -> set:
    body = _stem(text, strip)
    if len(body) < 2:
        return {body} if body else set()
    return {body[i:i + 2] for i in range(len(body) - 1)}


_YEAR = re.compile(r"(?:19|20)\d{2}")

# 不承载业务口径的通用词：虚词、疑问词、量词、常见的"问法外壳"。
# 抽"实义关键词"前先把它们替换成空格，剩下的片段才有可能是口径线索（如"大单"）。
# 原则：宁可漏抽（退回字面相似度），也不要多抽——多抽会让两个无关问题共享一个假关键词。
_GENERIC_WORDS = (
    "是多少", "有多少", "查一下", "看一下",
    "一共", "总共", "合计", "多少", "几个", "几笔", "几条", "哪几", "哪些", "什么",
    "怎么", "如何", "为什么", "主要", "来自", "分布", "情况", "分别", "统计",
    "请问", "帮我", "看看", "占比", "排名", "对比", "变化", "趋势", "明细", "列表",
    "一下", "这些", "那些", "所有", "全部", "每个", "各个", "多少笔", "多少钱",
    # 通用实体与动词：库里都有对应概念，谁问都会带上，不是口径线索。
    # 不排除它们的话，"成交了多少笔"会与"大单"那条记录共享"成交"而被误判为同类。
    "成交", "销售", "购买", "下单", "交易", "消费", "采购", "产生", "发生",
    "商品", "产品", "订单", "客户", "数据", "记录", "信息", "业务",
    # 单字虚词/量词：既要去掉，也充当分隔符——否则"华南的大单"会连成一个片段，
    # 抽不出"大单"这个真正的口径线索。
    "的", "了", "是", "和", "与", "及", "或", "在", "有", "个", "笔", "元",
    "各", "每", "这", "那", "都", "也", "就", "最", "再", "又", "只", "才",
    "把", "被", "给", "对", "从", "到", "向", "跟", "同", "按", "吗", "呢", "吧",
)

# 命中"库外实义词"时的相关度：比字面相似更强，因为那是同一套自定义口径
# （如两个问题都提到"大单"），维度不同只是同一口径的不同切法，不影响它适用。
SIGNATURE_SCORE = 0.8


def _stem(text: str, strip=()) -> str:
    """去掉标点、通用词与维度别名后剩下的"骨架"串（去空格）。

    字面相似度必须算在骨架上，不能算在原句上：「成交了多少笔」和
    「一共成交了多少大单」的原句有 0.44 的相似度，共享的却全是
    "成交/了多少"这种问法外壳——真正的区别是"大单"，而那恰恰被相似度模糊掉了。

    strip 传维度别名（"地区""品类"这类）：它们是切片的标签而不是口径，
    「华南地区的销售额」去掉"地区"后才和「华南的销售额」对上。
    """
    body = _question_key(text)
    for word in sorted(set(_GENERIC_WORDS) | set(strip), key=len, reverse=True):
        body = body.replace(word, " ")
    return "".join(body.split())


def _keywords(text: str) -> set:
    """抽出实义片段（去标点、去通用词后长 ≥2 的片段）。"""
    body = _question_key(text)
    for word in sorted(_GENERIC_WORDS, key=len, reverse=True):
        body = body.replace(word, " ")
    return {part for part in body.split() if len(part) >= 2}


def dimension_aliases(enums: Optional[list]) -> tuple:
    """维度别名：枚举列的 label 与线索词（"地区""品类""等级"…）。

    它们只说明"按哪个切片看"，不承载口径，比对骨架与判相关度时都该忽略。
    """
    words = []
    for enum in enums or []:
        label = str(enum.get("label") or "").strip()
        if label:
            words.append(label)
        words.extend(str(h) for h in (enum.get("hints") or []) if h)
    return tuple(words)


def known_words(enums: Optional[list], terms: Optional[dict] = None) -> set:
    """库内已知的业务词：术语名 + 字段标签 + 枚举取值。

    这些词的含义由 schema / 术语表说了算，不需要靠修正记录解释。
    **不在其中的实义词**才是口径线索——"大单"没在术语表里，说明它是用户
    自己约定的口径，只能靠 lesson 传下去。
    """
    words = {str(k) for k in (terms or {})}
    for enum in enums or []:
        words.update(str(v) for v in (enum.get("values") or []) if v)
    words.update(dimension_aliases(enums))     # "地区""品类"这类切片标签也不算口径
    return words


def signature_words(text: str, known: Optional[set] = None) -> set:
    """库外实义词：`_keywords(text) - known`。"""
    return _keywords(text) - set(known or ())


def _dimension_groups(enums: Optional[list]) -> List[dict]:
    """把枚举列整理成"维度组"，用于判断两个问题是不是在说同一个切片。

    每组 = 一个枚举列：真实取值 + 中文标签 + 线索词。
    换库后自动适配（取值来自数据画像），不写死任何业务词。
    """
    groups = []
    for enum in enums or []:
        values = {str(v) for v in (enum.get("values") or []) if v}
        words = set()
        for key in ("column", "label"):
            word = str(enum.get(key) or "").strip()
            if word:
                words.add(word)
        words.update(str(h) for h in (enum.get("hints") or []) if h)
        if values or words:
            groups.append({"name": enum.get("column") or "", "values": values,
                           "words": words})
    return groups


def _match_dimensions(text: str, groups: List[dict]) -> List[tuple]:
    """问题命中了哪些维度：[(维度名, 命中到的具体取值集合)]。

    具体取值（"华南"）与泛指（"各区域"）都算命中同一维度，但取值集合不同——
    这个区别正是"能不能套用这条示范"的关键。
    """
    hits = []
    for group in groups:
        values = {v for v in group["values"] if v in text}
        generic = any(w in text for w in group["words"] if w not in group["values"])
        if values or generic:
            hits.append((group["name"], values))
    return hits


def score_lesson(question: str, lesson: dict, enums: Optional[list] = None,
                 terms: Optional[dict] = None) -> float:
    """相关度：口径词命中 > 字面相似 × 条件一致性惩罚。

    只靠字面相似会两头出错：
    - **误命中**：实测「各区域的销售额」与「各品类的销售额」对同一条「华南的销售额」
      都打出 0.67 —— 共享"的销售额"三个字就够了。可真把 `WHERE region='华南'`
      的示范塞给"各品类"的问题，模型很容易照抄那个条件；
    - **漏命中**：「大单主要来自哪些品类」对「一共成交了多少大单」只有 0.05，
      明明是同一套口径（大单 = 金额 ≥ 10000），却一个字都传不过去。

    所以先看**库外实义词**（"大单"这种不在术语表里的词）：两个问题共享它，
    就是在问同一套自定义口径，直接给高分——维度不同反而是"同一口径的另一种切法"，
    正是该把写法传下去的场景。没有共享口径词时，才退回字面相似度，并加三道硬判断：
    1. 两边都指定了具体取值但没有交集 → 0；
    2. 两边提到的是不同维度（区域 vs 品类）→ 0；
    3. 一边比另一边多了限定（泛指 vs 具体、带年份 vs 不带）→ 打 0.6 折，
       宁可少给一条示范，也不给一条条件对不上的。
    """
    lesson_question = lesson.get("question", "")
    groups = _dimension_groups(enums)
    q_hits, l_hits = _match_dimensions(question, groups), _match_dimensions(lesson_question, groups)

    q_values = set().union(*[v for _, v in q_hits]) if q_hits else set()
    l_values = set().union(*[v for _, v in l_hits]) if l_hits else set()

    known = known_words(enums, terms)
    shared = signature_words(question, known) & signature_words(lesson_question, known)
    aliases = dimension_aliases(enums)
    if shared and not (q_values and l_values and not (q_values & l_values)):
        # 共享口径词：同一套业务定义。取值冲突（华南 vs 华东）仍然拦住，
        # 否则会把"限定华南"的示范套到"华东"的问题上。
        return round(min(0.95, SIGNATURE_SCORE + 0.05 * len(shared)), 4)

    a, b = _bigrams(question, aliases), _bigrams(lesson_question, aliases)
    if not a or not b:
        return 0.0
    base = len(a & b) / len(a | b)

    if q_values and l_values and not (q_values & l_values):
        return 0.0                       # 华南 vs 华东：条件冲突
    q_dims = {name for name, _ in q_hits}
    l_dims = {name for name, _ in l_hits}
    if q_dims and l_dims and not (q_dims & l_dims):
        return 0.0                       # 区域 vs 品类：不是同一个切片

    penalty = 1.0
    if bool(q_values) != bool(l_values):
        penalty *= 0.6                   # 一边具体一边泛指
    if bool(q_dims) != bool(l_dims):
        penalty *= 0.6                   # 一边提了切片一边没提

    q_years, l_years = set(_YEAR.findall(question)), set(_YEAR.findall(lesson_question))
    if q_years and l_years and not (q_years & l_years):
        return 0.0                       # 2024 vs 2025：换了一年
    if bool(q_years) != bool(l_years):
        penalty *= 0.75

    bonus = 0.0
    if q_values & l_values:
        bonus = VALUE_BONUS * min(len(q_values & l_values), 2) / 2
    return round(min(1.0, (base + bonus) * penalty), 4)


def select_lessons(question: str, lessons: List[dict], enums: Optional[list] = None,
                   terms: Optional[dict] = None,
                   limit: int = MAX_LESSONS_IN_PROMPT) -> List[dict]:
    """挑出与当前问题相关的 lesson（最多 limit 条，按相关度降序）。

    enums 是数据画像里的枚举列，terms 是业务术语表——两者一起用来判断
    "是不是在说同一个切片、是不是同一套口径"。
    """
    if not lessons or not question:
        return []

    scored = []
    for lesson in lessons:
        score = score_lesson(question, lesson, enums, terms)
        if score >= MIN_SCORE:
            scored.append((score, lesson))
    scored.sort(key=lambda pair: (pair[0], pair[1].get("created") or ""), reverse=True)

    picked, used = [], 0
    for _, lesson in scored[:limit]:
        cost = len(lesson.get("output") or "") + len(lesson.get("question") or "")
        if picked and used + cost > MAX_INJECT_CHARS:
            break            # 预算用完就少给一条，不截断 lesson（截断的 SQL 会教坏模型）
        picked.append(lesson)
        used += cost
    return picked
