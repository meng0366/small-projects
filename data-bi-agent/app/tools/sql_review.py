"""SQL 结构审查（第十轮 P1，对应 F2.14）。

安全检查（`sql_tool.check_sql_safety`）管的是"能不能执行"，
时段/取值守卫（`scope_guard`）管的是"有没有偷偷改用户的条件"，
这里补第三类：**SQL 写出来了、也能跑，但结构本身会把数字算错**。

三类问题，按代价分级：
- **block**：笛卡尔积（逗号连表没有关联条件、JOIN 缺 ON、ON 恒真）。
  这类查询行数相乘，结果必然不可信，且可能把 CPU 打满 —— 不给执行，
  把原因回喂给模型改写（保持 SQL 路线，不换 pandas）。
- **warn**：`SELECT *`、非唯一键关联、显式 CROSS JOIN。
  这些结果**可能**是对的（例如先 GROUP BY 的一对多关联），拦错比漏判更伤，
  所以只提示、不拦截，让用户能看见风险。

判据全部是纯函数 + 代码，不花 token。唯一性来自
`schema.loader.describe_unique_keys`（主键 / UNIQUE / catalog 的 ref 与多对一声明），
拿不到唯一性信息时宁可不报warn —— 不猜"看起来像 ID 的列就是唯一的"。

词法处理刻意不引入 sqlparse：项目定位是轻量 MVP，只需在**顶层**识别几个子句，
用一层括号深度 + 字符串/注释遮蔽就够了；子查询内部的 JOIN 不审查（避免误报）。
"""
import re

BLOCK = "block"
WARN = "warn"

_MESSAGES = {
    "cartesian": "FROM 里有多张表却没有关联条件（笛卡尔积），行数会相乘、汇总会失真；"
                 "请改成显式 JOIN ... ON 或在 WHERE 里写清两表关联。",
    "join_no_on": "JOIN 缺少 ON / USING 关联条件，会产生笛卡尔积；请补上两表关联。",
    "join_always_true": "JOIN 的 ON 条件是恒真的（如 1 = 1），等价于笛卡尔积；请写真实的关联列。",
    "select_star": "SELECT * 会返回全部列（列名可能重复、下游也难核对）；请只列出问题中需要的列。",
    "non_unique_join": "JOIN 两侧都不是唯一键，可能一对多放大行数、把金额重复计数；"
                       "请确认关联粒度，改用唯一键关联或先聚合。",
    "cross_join": "CROSS JOIN 会做全组合、行数相乘；如果不是刻意求全部组合，请改成带 ON 的 JOIN。",
}

# 字符串/注释先遮蔽，避免把 `'SELECT *'` 或注释里的 JOIN 当成真语句。
_MASK_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|--[^\n]*|/\*.*?\*/",
    re.DOTALL,
)

_EVENT_RE = re.compile(
    r"\(|\)|\bCROSS\s+JOIN\b|\bNATURAL\s+JOIN\b"
    r"|\b(?:LEFT|RIGHT|INNER|FULL|OUTER)\s+(?:OUTER\s+)?JOIN\b|\bJOIN\b"
    r"|\bON\b|\bUSING\b|\bSELECT\b|\bFROM\b|\bWHERE\b|\bGROUP\s+BY\b"
    r"|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|\bOFFSET\b|\bUNION\b|\bWITH\b",
    re.IGNORECASE,
)

_COL_PAIR_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\b",
    re.IGNORECASE,
)

_TABLE_REF_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(?:AS\s+([A-Za-z_]\w*)|([A-Za-z_]\w*))?", re.IGNORECASE)

_STOP_AT = {"JOIN", "ON", "USING", "WHERE", "GROUP BY", "ORDER BY",
            "HAVING", "LIMIT", "OFFSET", "UNION", "WITH"}
_SQL_WORDS = {"select", "values", "where", "on", "join", "set", "union", "with"}


def _mask(sql: str) -> str:
    """把字符串字面量与注释替换成等长空格（保留换行，便于调试）。"""
    chars = list(sql or "")
    for match in _MASK_RE.finditer(sql or ""):
        for pos in range(match.start(), match.end()):
            if chars[pos] != "\n":
                chars[pos] = " "
    return "".join(chars)


def _events(masked: str) -> list:
    """顶层子句事件 [(start, end, kind)]；括号里的子查询不计入。"""
    result = []
    depth = 0
    for match in _EVENT_RE.finditer(masked):
        token = re.sub(r"\s+", " ", match.group()).upper()
        if token == "(":
            depth += 1
            continue
        if token == ")":
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if token.endswith("JOIN"):
            token = "JOIN" if token in ("LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
                                        "FULL JOIN", "OUTER JOIN",
                                        "LEFT OUTER JOIN", "RIGHT OUTER JOIN") else token
        result.append((match.start(), match.end(), token))
    return result


def _next(events: list, index: int, kinds: set) -> int:
    """从 index 之后找第一个指定类型的事件位置，找不到返回 len(events)。"""
    for j in range(index + 1, len(events)):
        if events[j][2] in kinds:
            return j
    return len(events)


def _split_top(text: str) -> list:
    """按顶层逗号切分，忽略括号与函数参数里的逗号。"""
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _table_ref(segment: str):
    """从一段表引用里取 (表名, 别名)；识别不出来返回 ('', '')。"""
    match = _TABLE_REF_RE.match(segment)
    if not match:
        return "", ""
    table = match.group(1)
    if table.lower() in _SQL_WORDS:
        return "", ""
    alias = match.group(2) or match.group(3) or table
    return table, alias


def _col_pairs(text: str) -> list:
    return [(m.group(1), m.group(2), m.group(3), m.group(4))
            for m in _COL_PAIR_RE.finditer(text)]


def _is_unique(unique_keys: dict, aliases: dict, table: str, column: str) -> bool:
    real = aliases.get(table, table)
    return column.lower() in {str(c).lower() for c in (unique_keys.get(real) or [])}


def _issue(code: str, issues: list) -> None:
    """同一类问题只报一次：给模型的反馈越短越容易照着改。"""
    if all(item["code"] != code for item in issues):
        issues.append({"code": code, "level": BLOCK if code in
                       ("cartesian", "join_no_on", "join_always_true") else WARN,
                       "message": _MESSAGES[code]})


def review_sql(sql: str, unique_keys: dict = None) -> dict:
    """审查一条 SQL 的结构，返回 {"ok", "issues", "blocks", "warns"}。

    `unique_keys` 形如 `{表名: {列名}}`，来自 `describe_unique_keys`；
    传 None 时只检查与唯一性无关的结构问题（笛卡尔积 / SELECT *）。
    """
    masked = _mask(sql)
    events = _events(masked)
    issues = []
    unique_keys = unique_keys or {}
    aliases = {}

    # ---- 1. FROM 段：逗号连表且没有关联条件 = 笛卡尔积 ----
    from_index = next((i for i, e in enumerate(events) if e[2] == "FROM"), None)
    if from_index is not None:
        end = _next(events, from_index, _STOP_AT)
        segment = masked[events[from_index][1]: events[end][0] if end < len(events) else None]
        if "(" not in segment:  # 子查询/派生表解析不可靠，宁可不判
            sources = []
            for part in _split_top(segment):
                table, alias = _table_ref(part)
                if table:
                    sources.append(alias)
                    aliases[alias] = table
            if len(sources) >= 2:
                conditions = []
                for i, event in enumerate(events):
                    if event[2] not in ("ON", "WHERE"):
                        continue
                    stop = _next(events, i, _STOP_AT - {"ON"})
                    conditions.append(masked[event[1]: events[stop][0]
                                             if stop < len(events) else None])
                linked = any(a in sources and b in sources and a != b
                             for text in conditions
                             for a, _, b, _ in _col_pairs(text))
                if not linked:
                    _issue("cartesian", issues)

    # ---- 2. SELECT 段：SELECT * ----
    select_index = next((i for i, e in enumerate(events) if e[2] == "SELECT"), None)
    if select_index is not None:
        end = _next(events, select_index, {"FROM", "UNION", "WHERE", "GROUP BY",
                                           "ORDER BY", "LIMIT"})
        segment = masked[events[select_index][1]: events[end][0]
                         if end < len(events) else None]
        for field in _split_top(segment):
            if re.fullmatch(r"(?:DISTINCT\s+)?(?:[A-Za-z_]\w*\s*\.\s*)?\*",
                            field.strip(), re.IGNORECASE):
                _issue("select_star", issues)
                break

    # ---- 3. JOIN 段：缺 ON / 恒真 ON / 两侧都不唯一 ----
    for i, event in enumerate(events):
        if event[2] not in ("JOIN", "CROSS JOIN", "NATURAL JOIN"):
            continue
        if event[2] in ("CROSS JOIN", "NATURAL JOIN"):
            _issue("cross_join", issues)
        stop = _next(events, i, _STOP_AT)
        table_segment = masked[event[1]: events[stop][0] if stop < len(events) else None]
        table, alias = _table_ref(table_segment)
        if table:
            aliases[alias] = table
        if event[2] != "JOIN":
            continue
        # 语句到末尾都没有 ON/USING 也是"缺关联条件"，不能因为 stop 越界就跳过
        kind = events[stop][2] if stop < len(events) else ""
        if kind == "USING":
            continue
        if kind != "ON":
            _issue("join_no_on", issues)
            continue
        on_end = _next(events, stop, _STOP_AT - {"ON"})
        on_text = masked[events[stop][1]: events[on_end][0]
                         if on_end < len(events) else None]
        if re.search(r"\b1\s*=\s*1\b", on_text) or re.search(r"\bTRUE\b", on_text, re.I):
            _issue("join_always_true", issues)
        pairs = _col_pairs(on_text)
        if pairs and unique_keys:
            risky = all(
                not _is_unique(unique_keys, aliases, t1, c1)
                and not _is_unique(unique_keys, aliases, t2, c2)
                for t1, c1, t2, c2 in pairs
            )
            if risky:
                _issue("non_unique_join", issues)

    blocks = sum(1 for item in issues if item["level"] == BLOCK)
    return {"ok": blocks == 0, "issues": issues, "blocks": blocks,
            "warns": len(issues) - blocks}


def format_review_note(issues: list) -> str:
    """给用户看的提示：只展示 warn（block 根本不会执行）。"""
    warns = [item["message"] for item in issues if item["level"] == WARN]
    return "结构审查提示：" + "；".join(warns) if warns else ""


def format_review_for_model(issues: list) -> str:
    """回喂给模型的改写要求：说清结构错在哪，别顺手改用户的口径。"""
    lines = [f"- {item['message']}" for item in issues if item["level"] == BLOCK]
    if not lines:
        return ""
    return ("SQL 结构审查未通过（没有执行这条语句）：\n" + "\n".join(lines) +
            "\n请保持用户问题的口径、时间与筛选条件不变，只改写 SQL 结构后重试。")
