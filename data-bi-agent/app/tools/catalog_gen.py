"""描述自动生成（P1，2026-11-25）：从一个陌生的库生成 catalog 草稿。

为什么需要它
------------
`catalog.yaml` 是手写的。接一个新库，用户得先搞清楚每张表每列是什么意思，
再照着 YAML 格式抄一遍 —— 这恰恰是"接库成本"里最劝退的一步，也是模型最擅长的部分。
但反过来，**事实绝不能问模型**：让它看 `discount` 列的样例，它会写 0.9，
而库里实际是 0.85 ~ 1.0；让它列枚举取值，它会凭常识补一个库里没有的"华东区"。

所以分工是：

| 谁 | 负责 |
| --- | --- |
| 代码 | 事实：类型映射、样例值、枚举取值、主外键推断、行数与空值 |
| 模型 | 语义：中文表名/列名、一句话说明、提问线索词（hints）、指标口径建议 |
| 人 | 拍板：输出的是 `catalog.draft.yaml`，带 TODO 标记，改名才生效 |

为什么不直接覆盖 catalog.yaml
------------------------------
词典是"已确认的口径"。自动生成直接覆盖，等于让一次模型输出把人工校准过的定义洗掉 ——
而词典恰恰是全局最不该被自动改写的东西。所以默认写草稿；`--merge` 也只做**补空**
（已有非空字段一律不动），永远不删已有内容。

枚举列怎么认
------------
只靠"去重个数少"会错得很难看：客户表 20 行，`customer_name` 也是 20 个不同值，
但它是标识符不是枚举。这里用四条判据叠加：

1. 不是主键、去重个数在 2..20 之间；
2. 去重数 < 总行数 × 0.8（每行都不同 → 是标识符）；
3. 文本列直接算候选，日期列排除；
4. 数值列只在列名像状态/类型/等级时才算 —— 否则 `quantity`（1..10）会被当成枚举。

宁可漏认：漏认的后果是模型看不到取值（退回两段式探查能补），
误认的后果是把 `quantity` 的 1..10 写进"必须严格使用以下字面量"，那会主动制造错误。
"""
import json
import os
import re
import sqlite3
from typing import Any, Optional

from app.schema.loader import list_tables

ENUM_MAX_VALUES = 20
"""单列最多算多少个取值还算枚举（与 loader.MAX_ENUM_VALUES 对齐）。"""

SAMPLE_VALUES = 3
"""每列取几个样例值（取最常见的，不取最小的 —— 前几行往往是 NULL 或占位值）。"""

SAMPLE_MAX_CHARS = 40
"""样例值 / 取值的字符上限，防止备注类长文本把提示词撑爆。"""

FK_MAX_TABLES = 40
"""超过这个表数就跳过外键推断。推断是"每列 × 每表"的值域比对，大库上不划算。"""

ENUM_RATIO = 0.8
"""去重数 / 总行数 超过这个比例，视为"每行都不同"的标识符，不是枚举。"""

DATE_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")
"""按值认日期：SQLite 里日期普遍以 TEXT 存，只看声明类型会把它们全当成字符串。"""

ENUM_NAME_HINT = re.compile(
    r"(status|state|type|kind|level|grade|stage|flag|category|class|channel|"
    r"source|method|方式|状态|类型|级别|渠道|来源|类别|等级)", re.I)
"""数值列只有当名字像"状态/类型/等级"时才当枚举候选（`quantity` 会被排除）。"""

IDENTIFIER_NAME = re.compile(
    r"^(id|uuid|guid)$|(_id|_uuid|_guid|_no|_code|_number|_key|id)$", re.I)
"""标识符命名：这些列是"某一个具体的东西"，不是"某一类"，永远不该当枚举。

`orders.customer_id` 就是活例子：3877 行里 20 个不同值、是文本列，
按"去重少 + 文本"本来会被认成枚举，然后 schema 里就会出现
「customer_id 必须严格使用以下字面量：C001 / C002 / ...」—— 主动制造错误。
宁可漏认（`status_code` 这类会一起被排除，少一次取值约束），不能误认。"""


class CatalogError(Exception):
    """无法生成词典（库不可读、没有表等）。"""


# ---------------------------------------------------------------- 基础工具

def _q(name: str) -> str:
    """SQLite 标识符加引号（表名/列名来自库本身，仍要转义双引号）。"""
    return '"' + str(name).replace('"', '""') + '"'


def _shorten(value: Any, limit: int = SAMPLE_MAX_CHARS) -> str:
    text = str(value).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _num_text(value: Any) -> str:
    """数字转文本：3801.0 → 3801，0.95 → 0.95（给人看的口径，不带多余小数位）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    return str(value)


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


# ---------------------------------------------------------------- 类型映射

def looks_like_dates(values) -> bool:
    values = [str(v).strip() for v in (values or []) if v is not None and str(v).strip()]
    return bool(values) and all(DATE_VALUE_RE.match(v) for v in values)


def infer_type_from_values(values) -> str:
    """完全没声明类型时（`CREATE TABLE ... AS SELECT` 很常见），只能看值。"""
    values = [str(v).strip() for v in (values or []) if v is not None and str(v).strip()]
    if not values:
        return "string"
    if looks_like_dates(values):
        return "date"
    numeric = [v for v in values if re.fullmatch(r"[-+]?\d+(\.\d+)?", v)]
    if len(numeric) == len(values):
        return "int" if all(re.fullmatch(r"[-+]?\d+", v) for v in values) else "float"
    return "string"


def map_type(raw_type: str, values=None) -> str:
    """SQLite 声明类型 → catalog 的 type（string / int / float / date / bool）。

    TEXT 也要看值：很多库把日期存成 TEXT，声明类型看不出区别，
    而"是不是日期列"直接决定 `collect_data_profile` 会不会去算时间覆盖范围
    （schema 里少这一段，模型就不知道"2024 年没数据"）。
    """
    text = (raw_type or "").strip().upper()
    if "BOOL" in text:
        return "bool"
    if "INT" in text:
        return "int"
    if any(k in text for k in ("REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL", "MONEY")):
        return "float"
    if any(k in text for k in ("DATE", "TIME")):
        return "date"
    if any(k in text for k in ("CHAR", "CLOB", "TEXT", "STRING")):
        return "date" if looks_like_dates(values) else "string"
    if text:
        return "string"
    return infer_type_from_values(values)


def is_enum_candidate(column: str, col_type: str, distinct: int, total: int,
                      is_key: bool = False) -> bool:
    """四条判据叠加（见模块注释）。宁可漏认，不能误认。"""
    if is_key or total <= 0 or distinct < 2 or distinct > ENUM_MAX_VALUES:
        return False
    if IDENTIFIER_NAME.search(str(column)):
        return False
    if distinct >= total * ENUM_RATIO:
        return False
    if col_type == "date":
        return False
    if col_type == "string":
        return True
    return bool(ENUM_NAME_HINT.search(str(column)))


# ---------------------------------------------------------------- 实测画像

def _column_stats(con: sqlite3.Connection, table: str, column: str) -> dict:
    q, tq = _q(column), _q(table)
    row = con.execute(f"SELECT COUNT(*), COUNT({q}), COUNT(DISTINCT {q}) FROM {tq}").fetchone()
    total, nonnull, distinct = int(row[0]), int(row[1]), int(row[2])
    # 取最常见的 3 个值当样例：前 3 行常是 NULL 或占位值，说明不了这列是什么
    samples = con.execute(
        f"SELECT {q} FROM {tq} WHERE {q} IS NOT NULL "
        f"GROUP BY {q} ORDER BY COUNT(*) DESC, {q} LIMIT {SAMPLE_VALUES}"
    ).fetchall()
    return {"total": total, "nonnull": nonnull, "distinct": distinct,
            "raw": [r[0] for r in samples]}


def _distinct_list(con: sqlite3.Connection, table: str, column: str,
                   limit: int = ENUM_MAX_VALUES) -> list:
    rows = con.execute(
        f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
        f"WHERE {_q(column)} IS NOT NULL ORDER BY 1 LIMIT {limit}"
    ).fetchall()
    return [r[0] for r in rows]


def _value_text(value: Any) -> str:
    """给样例值用的文本：浮点保留 `1.0` 这样的小数位。

    为什么不复用 `_num_text`：`discount` 的样例写 `1` 会被读成"整数 1"，
    而它实际是"1.0 表示无折扣"；范围（`0.9 ~ 1`）去掉多余小数更好读，
    样例（`example: 1.0`）保留更准确 —— 两处需求相反，所以分成两个函数。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _typed(value: Any, col_type: str) -> Any:
    """样例值保留原本的数字类型（`example: 3999` 而不是 `"3999"`），其余转短字符串。"""
    if col_type == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    if col_type == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return _shorten(value)


def _declared_fks(con: sqlite3.Connection, table: str, column_map: dict) -> list:
    """读建表时声明的外键，返回 [(from_col, to_table, to_col), ...]。"""
    try:
        rows = con.execute(f"PRAGMA foreign_key_list({_q(table)})").fetchall()
    except sqlite3.Error:
        return []
    out = []
    for r in rows:
        to_table, from_col, to_col = r[2], r[3], r[4]
        if not from_col or not to_table:
            continue
        if not to_col or to_col == "None":
            # 省略目标列时指向目标表主键
            pk = [c for c, meta in (column_map.get(to_table) or {}).items() if meta.get("pk")]
            to_col = pk[0] if pk else ""
        if to_col:
            out.append((from_col, to_table, to_col))
    return out


def _is_unique_col(con: sqlite3.Connection, table: str, column: str) -> bool:
    row = con.execute(
        f"SELECT COUNT({_q(column)}), COUNT(DISTINCT {_q(column)}) FROM {_q(table)}"
    ).fetchone()
    return row[0] > 1 and row[0] == row[1]


def _is_subset(con: sqlite3.Connection, src_table: str, src_col: str,
               dst_table: str, dst_col: str) -> bool:
    """源列的值是否全部出现在目标列里（EXCEPT 为空）。

    SQLite 弱类型，`'1'` 与 `1` 会被当成不同值 —— 误判方向是"漏认外键"，
    比"错认外键"安全（错认会让 schema 里出现一条假关系，模型照着 JOIN 就错了）。
    """
    sql = (f"SELECT COUNT(*) FROM ("
           f"SELECT DISTINCT {_q(src_col)} FROM {_q(src_table)} "
           f"WHERE {_q(src_col)} IS NOT NULL "
           f"EXCEPT SELECT DISTINCT {_q(dst_col)} FROM {_q(dst_table)} "
           f"WHERE {_q(dst_col)} IS NOT NULL)")
    return con.execute(sql).fetchone()[0] == 0


def profile_schema(db_path: str) -> dict:
    """实测库结构：表、列、类型、样例、枚举候选、主外键。零 token，可反复跑。

    返回：
        {"db": 路径,
         "tables": [{"name", "rows", "columns": [
             {"name","type","raw_type","distinct","nulls","examples","values","pk","fk","range"}
         ]}],
         "relationships": [{"from_table","from_column","to_table","to_column","source"}],
         "notes": [str]}
    """
    profile = {"db": db_path, "tables": [], "relationships": [], "notes": []}
    if not db_path or not os.path.exists(db_path):
        profile["notes"].append(f"数据库文件不存在：{db_path}")
        return profile

    con = _connect(db_path)
    try:
        tables = list_tables(db_path)
        if not tables:
            profile["notes"].append("库里没有用户表")
            return profile

        column_map = {}          # {表名: {列名: 列元信息}}，供外键推断用
        for table in tables:
            try:
                info = con.execute(f"PRAGMA table_info({_q(table)})").fetchall()
                rows = con.execute(f"SELECT COUNT(*) FROM {_q(table)}").fetchone()[0]
            except sqlite3.Error as e:
                profile["notes"].append(f"跳过表 {table}：{e}")
                continue
            if not info:
                continue

            cols, entries = {}, []
            for _, name, raw_type, _, _, pk in info:
                stats = _column_stats(con, table, name)
                col_type = map_type(raw_type, stats["raw"])
                entry = {
                    "name": name,
                    "type": col_type,
                    "raw_type": (raw_type or "").strip(),
                    "distinct": stats["distinct"],
                    "nulls": stats["total"] - stats["nonnull"],
                    "examples": [_typed(v, col_type) for v in stats["raw"]],
                    "values": [],
                    "pk": bool(pk),
                    "fk": "",
                    "range": "",
                }
                if col_type in ("int", "float"):
                    mn, mx = con.execute(
                        f"SELECT MIN({_q(name)}), MAX({_q(name)}) FROM {_q(table)}"
                    ).fetchone()
                    if mn is not None and mn != mx:
                        entry["range"] = f"{_num_text(mn)} ~ {_num_text(mx)}"
                if is_enum_candidate(name, col_type, stats["distinct"],
                                     stats["total"], bool(pk)):
                    values = _distinct_list(con, table, name)
                    entry["values"] = [str(v) for v in values]
                entries.append(entry)
                cols[name] = entry

            column_map[table] = cols
            profile["tables"].append({"name": table, "rows": int(rows), "columns": entries})

        _collect_relationships(con, profile, column_map, len(tables))
    except sqlite3.Error as e:
        # 不是数据库、被占用、权限不足：清空半份结果，让调用方按"没有可读的表"处理，
        # 而不是拿着一半表去生成一份看起来正常的草稿
        profile["notes"].append(f"读取库失败：{e}")
        profile["tables"] = []
        profile["relationships"] = []
    finally:
        con.close()
    return profile


def _add_relationship(profile: dict, from_table: str, from_column: str,
                      to_table: str, to_column: str, source: str,
                      column_map: dict) -> None:
    if any(r["from_table"] == from_table and r["from_column"] == from_column
           for r in profile["relationships"]):
        return
    target = (column_map.get(to_table) or {}).get(to_column) or {}
    # 目标列唯一 → 一对一；否则多对一
    one_to_one = bool(target.get("pk"))
    profile["relationships"].append({
        "from_table": from_table, "from_column": from_column,
        "to_table": to_table, "to_column": to_column, "source": source,
        "cardinality": "一对一" if one_to_one else "多对一",
    })
    entry = (column_map.get(from_table) or {}).get(from_column)
    if entry is not None and not entry.get("fk"):
        entry["fk"] = f"{to_table}.{to_column}"


def _collect_relationships(con, profile: dict, column_map: dict, table_count: int) -> None:
    """先取建表声明，再按"同名 + 目标唯一 + 值域包含"补推。

    为什么敢推：SQLite 里大量库根本没声明外键（`orders.customer_id` 只是个普通列），
    而"同名列 + 目标列唯一 + 源值域是目标的子集"这三条同时成立，基本只有外键一种解释。
    三条缺一不可 —— 少了值域校验，`order_id` 和 `customer_id` 会互相认成外键。
    """
    if table_count > FK_MAX_TABLES:
        profile["notes"].append(
            f"表太多（{table_count} 张，阈值 {FK_MAX_TABLES}），跳过外键推断，只保留建表时声明的外键")
        for table in column_map:
            for from_col, to_table, to_col in _declared_fks(con, table, column_map):
                _add_relationship(profile, table, from_col, to_table, to_col,
                                  "declared", column_map)
        return

    for table in column_map:
        for from_col, to_table, to_col in _declared_fks(con, table, column_map):
            if to_col not in (column_map.get(to_table) or {}):
                profile["notes"].append(
                    f"{table}.{from_col} 声明指向 {to_table}.{to_col}，但目标列不存在")
                continue
            _add_relationship(profile, table, from_col, to_table, to_col,
                              "declared", column_map)

    for table, cols in column_map.items():
        for col_name, entry in cols.items():
            if entry.get("fk") or entry.get("pk"):
                continue
            if not col_name.lower().endswith("_id") and col_name.lower() != "id":
                continue
            if entry.get("distinct", 0) < 2:
                continue
            for other_table, other_cols in column_map.items():
                if other_table == table or col_name not in other_cols:
                    continue
                if not other_cols[col_name].get("pk") and \
                        not _is_unique_col(con, other_table, col_name):
                    continue
                if not _is_subset(con, table, col_name, other_table, col_name):
                    continue
                _add_relationship(profile, table, col_name, other_table, col_name,
                                  "guessed", column_map)
                profile["notes"].append(
                    f"{table}.{col_name} 没有声明外键，按同名列 + 值域包含推断为 → "
                    f"{other_table}.{col_name}（请确认）")
                break


# ---------------------------------------------------------------- 给模型看的画像

def format_profile(profile: dict) -> str:
    """把画像排成模型能直接读的文本：每列一行，事实齐全但不冗余。"""
    tables = (profile or {}).get("tables") or []
    db = os.path.basename((profile or {}).get("db") or "")
    lines = [f"库文件：{db}（SQLite），共 {len(tables)} 张表", ""]

    for t in tables:
        lines.append(f"表 {t['name']}（{t['rows']} 行）")
        for c in t["columns"]:
            parts = [c["name"], c["raw_type"] or "（未声明类型）"]
            parts.append(f"{c['distinct']} 个不同值")
            if c["nulls"]:
                parts.append(f"{c['nulls']} 行为空")
            if c["examples"]:
                parts.append("样例 " + " / ".join(_value_text(v) for v in c["examples"]))
            if c["range"]:
                parts.append(f"范围 {c['range']}")
            if c["pk"]:
                parts.append("主键")
            if c["fk"]:
                parts.append(f"外键 → {c['fk']}")
            if c["values"]:
                parts.append("全部取值 " + " / ".join(c["values"]))
            lines.append("- " + " · ".join(parts))
        lines.append("")

    rels = (profile or {}).get("relationships") or []
    if rels:
        lines.append("已识别的表关系：")
        for r in rels:
            tag = "建表时声明" if r["source"] == "declared" else "推断，待确认"
            lines.append(f"- {r['from_table']}.{r['from_column']} → "
                         f"{r['to_table']}.{r['to_column']}（{r['cardinality']}，{tag}）")
        lines.append("")

    return "\n".join(lines).strip()


def summarize_profile(profile: dict) -> str:
    """一行摘要（命令行 dry-run 用）。"""
    tables = (profile or {}).get("tables") or []
    cols = sum(len(t["columns"]) for t in tables)
    enums = sum(1 for t in tables for c in t["columns"] if c["values"])
    rels = len((profile or {}).get("relationships") or [])
    return (f"{len(tables)} 张表 · {cols} 列 · {enums} 个枚举列 · {rels} 条表关系")


# ---------------------------------------------------------------- 模型调用

MAX_PROFILE_CHARS = 24000
"""画像文本超过这个长度就提示分批 —— 硬截断会让模型给出半份词典，比报错更误导。"""


def _default_invoke(system: str, user: str) -> str:
    """真正调模型。延迟 import：让本模块的纯函数部分不依赖 langchain，测试可离线跑。"""
    from app.llm import get_llm

    llm = get_llm()
    reply = llm.invoke([("system", system), ("human", user)])
    content = getattr(reply, "content", reply)
    return content if isinstance(content, str) else str(content)


def generate_descriptions(profile: dict, invoke=None) -> str:
    """让模型给出中文名、说明与指标口径建议。

    `invoke(system, user) -> str` 可注入，测试里换成固定回复即可离线跑。
    """
    from app.prompts.templates import CATALOG_USER_TEMPLATE, SYSTEM_CATALOG

    body = format_profile(profile)
    if len(body) > MAX_PROFILE_CHARS:
        raise CatalogError(
            f"库结构太大（画像 {len(body)} 字符，上限 {MAX_PROFILE_CHARS}）。"
            f"请先只接入部分表（VIEW 或子库），或用 --dry-run 看看自动识别结果。")
    user = CATALOG_USER_TEMPLATE.format(profile=body,
                                        table_count=len(profile.get("tables") or []))
    return (invoke or _default_invoke)(SYSTEM_CATALOG, user)


# ---------------------------------------------------------------- 模型输出的解析与校验

def parse_draft(text: str) -> Optional[dict]:
    """从模型输出里抠 JSON（取第一个 `{` 到最后一个 `}`，容忍 ```json 包裹）。"""
    raw = (text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate_draft(draft: dict, profile: dict) -> list:
    """核对模型返回的表/列/口径引用，返回问题清单（不修改入参）。

    三类问题：
    - `unknown_*`（error）：模型提到了不存在的表或列 —— 这类**必须丢掉**，
      否则词典里出现不存在的列，`build_db_schema_text` 会把它当真发下去；
    - `missing_column`（warn）：漏了列 —— 保留实测列，只是没中文名；
    - `bad_term`（warn）：口径引用了不存在的表列。
    """
    issues = []
    tables = {t["name"]: {c["name"] for c in t["columns"]}
              for t in (profile or {}).get("tables") or []}

    for item in (draft or {}).get("tables") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name not in tables:
            issues.append({"level": "error", "kind": "unknown_table",
                           "message": f"模型提到了库里没有的表：{name or '(空)'}"})
            continue
        given = item.get("columns") or {}
        if not isinstance(given, dict):
            given = {}
        for col in given:
            if col not in tables[name]:
                issues.append({"level": "error", "kind": "unknown_column",
                               "message": f"模型提到了 {name} 里没有的列：{col}"})
        for col in tables[name]:
            if col not in given:
                issues.append({"level": "warn", "kind": "missing_column",
                               "message": f"{name}.{col} 没有中文名，草稿里会留 TODO"})

    for term in (draft or {}).get("terms") or []:
        if not isinstance(term, dict):
            continue
        label = str(term.get("name") or "").strip()
        source = str(term.get("source") or "").strip()
        if not label:
            issues.append({"level": "warn", "kind": "bad_term",
                           "message": "模型给的指标口径缺少名称，已忽略"})
            continue
        if not term_source_exists(source, tables):
            issues.append({"level": "warn", "kind": "bad_term",
                           "message": f"口径「{label}」引用的 {source or '(空)'} 不存在，已忽略"})
    return issues


def term_source_exists(source: str, tables: dict) -> bool:
    """口径必须能指到真实列：`orders.amount` 或 `orders.amount/orders.order_id`。"""
    parts = [p for p in str(source or "").replace(" ", "").split("/") if p]
    if not parts:
        return False
    for part in parts:
        if "." not in part:
            return False
        table, column = part.split(".", 1)
        if column not in tables.get(table, set()):
            return False
    return True


def clean_hints(hints) -> list:
    """清洗模型给的线索词：去空、去重、限长度与条数。

    长度上限 8（"客户等级"是 4 个字，"会员消费等级"是 6 个），下限 1 ——
    `category` 手写版里就有一个单字的"类"，去掉会让"各类产品"识别不出来。
    """
    out = []
    for item in hints or []:
        text = str(item).strip()
        if not text or len(text) > 8 or text in out:
            continue
        out.append(text)
        if len(out) >= 6:
            break
    return out


def _pick(*candidates) -> str:
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


# ---------------------------------------------------------------- 合并成词典结构

def build_draft(profile: dict, llm: Optional[dict] = None,
                existing: Optional[dict] = None) -> dict:
    """把实测画像、模型提议、已有词典合成一份 draft（catalog 的结构）。

    优先级（谁说了算）：
    - 事实（type / values / 表关系）：**实测**。手写版可能已过期，实测不会；
    - 中文名与说明（label / description / note / hints）：已有 > 模型；
    - 样例值（example）：已有 > 实测 → 手写的更典型（amount 手写 7598.1，
      实测最常见可能是 3999，作为"长什么样"的示意不如一个真实订单金额）；
    - 指标口径（terms）：已有 > 模型，且模型的每条都要过 source 校验。
    """
    llm_tables = {}
    for item in (llm or {}).get("tables") or []:
        if isinstance(item, dict) and item.get("name"):
            llm_tables[str(item["name"]).strip()] = item

    prev_tables = {}
    for item in (existing or {}).get("tables") or []:
        if isinstance(item, dict) and item.get("name"):
            prev_tables[str(item["name"]).strip()] = item

    tables = []
    for t in (profile or {}).get("tables") or []:
        name = t["name"]
        meta, prev = llm_tables.get(name) or {}, prev_tables.get(name) or {}
        entry = {"name": name}
        label = _pick(prev.get("label"), meta.get("label"))
        if label:
            entry["label"] = label
        description = _pick(prev.get("description"), meta.get("description"))
        if description:
            entry["description"] = description

        llm_cols = meta.get("columns") if isinstance(meta.get("columns"), dict) else {}
        prev_cols = prev.get("columns") if isinstance(prev.get("columns"), dict) else {}

        columns = {}
        for c in t["columns"]:
            col_name = c["name"]
            lc = llm_cols.get(col_name) if isinstance(llm_cols.get(col_name), dict) else {}
            pc = prev_cols.get(col_name) if isinstance(prev_cols.get(col_name), dict) else {}
            item = {}
            col_label = _pick(pc.get("label"), lc.get("label"))
            if col_label:
                item["label"] = col_label
            item["type"] = c["type"]
            if pc.get("example") is not None:
                item["example"] = pc["example"]
            elif c["examples"]:
                item["example"] = c["examples"][0]
            if c["values"]:
                item["values"] = list(c["values"])
            # hints 是 enum_guard 定位"他在说哪一列"的唯一线索，只对枚举列有意义。
            # 已有的原样保留（可能人工调过），模型给的过一遍清洗
            if pc.get("hints"):
                item["hints"] = [str(h) for h in pc["hints"]]
            else:
                hints = clean_hints(lc.get("hints"))
                if hints and c["values"]:
                    item["hints"] = hints
            note = _pick(pc.get("note"), lc.get("note"))
            if note:
                item["note"] = note
            ref = _pick(pc.get("ref"), c.get("fk") or "")
            if ref:
                item["ref"] = ref
            columns[col_name] = item
        entry["columns"] = columns
        tables.append(entry)

    # 表关系：已有文本优先（可能带人工说明），否则由实测生成
    relationships = [str(r) for r in (existing or {}).get("relationships") or [] if str(r).strip()]
    if not relationships:
        for r in (profile or {}).get("relationships") or []:
            relationships.append(
                f"{r['from_table']}.{r['from_column']} -> {r['to_table']}.{r['to_column']} "
                f"({r['cardinality']})")

    # 指标口径：先放已有（顺序稳定），再补模型给的、能指到真实列的
    terms = dict((existing or {}).get("terms") or {})
    col_map = {t["name"]: {c["name"] for c in t["columns"]}
               for t in (profile or {}).get("tables") or []}
    for term in (llm or {}).get("terms") or []:
        if not isinstance(term, dict):
            continue
        term_name = str(term.get("name") or "").strip()
        definition = str(term.get("definition") or "").strip()
        if not term_name or not definition or term_name in terms:
            continue
        if not term_source_exists(str(term.get("source") or ""), col_map):
            continue
        terms[term_name] = definition

    return {"tables": tables, "relationships": relationships, "terms": terms}


def count_todos(draft: dict) -> int:
    """还有多少处需要人确认（缺中文名 / 模型自己写了"待确认"）。"""
    total = 0
    for t in (draft or {}).get("tables") or []:
        if not str(t.get("label") or "").strip() or "待确认" in str(t.get("label") or ""):
            total += 1
        if "待确认" in str(t.get("description") or ""):
            total += 1
        for meta in (t.get("columns") or {}).values():
            if not str(meta.get("label") or "").strip():
                total += 1
            elif "待确认" in str(meta.get("label") or ""):
                total += 1
            elif "待确认" in str(meta.get("note") or ""):
                total += 1
    return total


# ---------------------------------------------------------------- 渲染 YAML

BAD_PLAIN = re.compile(r"""[,:\[\]{}#&*!|>'"%@`\n\r\t]|^\s|\s$""")
AMBIGUOUS = re.compile(
    r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$"      # 数字样字符串 → 会被读成数字
    r"|^\d{4}-\d{2}-\d{2}"                           # 日期样 → 会被读成 date
    r"|^(true|false|yes|no|on|off|null|~)$", re.I)
"""不引号包裹就会被 YAML 解释成别的类型（数字/布尔/日期/null）。"""


def _is_plain_safe(text: str) -> bool:
    if not text or BAD_PLAIN.search(text):
        return False
    if AMBIGUOUS.match(text):
        return False
    return not text.startswith(("-", "?", "&", "*"))


def yaml_scalar(value: Any) -> str:
    """把一个值渲染成 YAML 标量。

    安全就不加引号（可读性优先）；否则用 JSON 双引号形式 —— JSON 字符串
    恰好是合法的 YAML 双引号标量，转义规则完全一致，比手写转义可靠。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _value_text(value)
    if isinstance(value, list):
        return "[" + ", ".join(yaml_scalar(v) for v in value) + "]"
    text = str(value)
    return text if _is_plain_safe(text) else json.dumps(text, ensure_ascii=False)


def _todo_note(meta: dict) -> str:
    label = str(meta.get("label") or "").strip()
    if not label:
        return "TODO 待补中文名"
    if "待确认" in label:
        return "TODO 待确认"
    if "待确认" in str(meta.get("note") or ""):
        return "TODO 待确认"
    return ""


FIELD_ORDER = ("label", "type", "example", "values", "hints", "note", "ref")
INLINE_MAX = 96


def _render_column(name: str, meta: dict, indent: int) -> list:
    keys = [k for k in FIELD_ORDER if k in meta]
    body = ", ".join(f"{k}: {yaml_scalar(meta[k])}" for k in keys)
    todo = _todo_note(meta)
    pad = " " * indent
    line = f"{pad}{yaml_scalar(name)}: {{ {body} }}"
    if len(line) + (len(todo) + 3 if todo else 0) <= INLINE_MAX:
        if todo:
            line += f"   # {todo}"
        return [line]
    out = [f"{pad}{yaml_scalar(name)}:"]
    for k in keys:
        out.append(f"{pad}  {k}: {yaml_scalar(meta[k])}")
    if todo:
        out.append(f"{pad}  # {todo}")
    return out


def render_catalog_yaml(draft: dict, source: str = "", generated_at: str = "",
                        notes: Optional[list] = None) -> str:
    """渲染成带注释的 catalog.yaml 草稿。

    格式与手写版一致（列少时单行 flow，列多时展开成 block），
    保证 `yaml.safe_load` 回来与 draft 完全等价 —— 这条有专门的往返用例兜着。
    """
    todos = count_todos(draft)
    lines = [
        "# 业务语义词典（catalog 草稿）",
        "# 由 `python -m app.cli catalog` 自动生成，**需要人工过一遍再启用**。",
        "#",
        "# 分工：类型 / 样例 / 枚举取值 / 表关系来自对库的实测；",
        "#      中文名、说明、指标口径由模型提议（口径已校验过能否指到真实列）。",
    ]
    if todos:
        lines.append(f"# 搜 TODO 找到需要你确认的地方（共 {todos} 处），改完覆盖 "
                     f"app/schema/catalog.yaml 即可生效。")
    else:
        lines.append("# 没有自动补不上的空缺，但**口径名**仍要你自己过一眼 ——")
        lines.append("# 词典里写「销售金额」而同事都问「销售额」，这个口径就命中不了。")
        lines.append("# 确认后覆盖 app/schema/catalog.yaml 即可生效。")
    if source:
        lines.append(f"# 数据源：{source}")
    if generated_at:
        lines.append(f"# 生成时间：{generated_at}")
    for note in notes or []:
        lines.append(f"# 注意：{note}")
    lines += ["", "tables:"]

    for t in (draft or {}).get("tables") or []:
        lines.append(f"  - name: {yaml_scalar(t['name'])}")

        label = str(t.get("label") or "").strip()
        if not label:
            lines.append('    label: ""   # TODO 待补中文表名')
        else:
            line = f"    label: {yaml_scalar(label)}"
            if "待确认" in label:
                line += "   # TODO 待确认"
            lines.append(line)

        description = str(t.get("description") or "").strip()
        if not description:
            lines.append('    description: ""   # TODO 待补一句话说明')
        else:
            line = f"    description: {yaml_scalar(description)}"
            if "待确认" in description:
                line += "   # TODO 待确认"
            lines.append(line)

        lines.append("    columns:")
        for col_name, meta in (t.get("columns") or {}).items():
            lines.extend(_render_column(col_name, meta, 6))
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    rels = (draft or {}).get("relationships") or []
    if rels:
        lines += ["", "relationships:"]
        lines.extend(f"  - {yaml_scalar(r)}" for r in rels)

    terms = (draft or {}).get("terms") or {}
    if terms:
        lines += ["", "terms:"]
        for name, definition in terms.items():
            lines.append(f"  {yaml_scalar(name)}: {yaml_scalar(definition)}")

    return "\n".join(lines) + "\n"


def describe_changes(draft: dict, existing: Optional[dict] = None) -> list:
    """草稿与已有词典的差异摘要（给人看，`--merge` 时尤其需要）。"""
    if not existing:
        tables = (draft or {}).get("tables") or []
        cols = sum(len(t.get("columns") or {}) for t in tables)
        terms = len((draft or {}).get("terms") or {})
        return [f"表 {len(tables)} 张 · 列 {cols} 个 · 指标口径 {terms} 条"]

    prev_tables = {t.get("name"): t for t in (existing or {}).get("tables") or []
                   if isinstance(t, dict)}
    added_tables, added_cols, kept = [], 0, 0
    for t in (draft or {}).get("tables") or []:
        prev = prev_tables.get(t["name"])
        if not prev:
            added_tables.append(t["name"])
            added_cols += len(t.get("columns") or {})
            continue
        prev_cols = prev.get("columns") or {}
        for col in (t.get("columns") or {}):
            if col in prev_cols:
                kept += 1
            else:
                added_cols += 1
    prev_terms = set((existing or {}).get("terms") or {})
    new_terms = [n for n in ((draft or {}).get("terms") or {}) if n not in prev_terms]

    out = []
    if added_tables:
        out.append(f"新增表：{' / '.join(added_tables)}")
    if added_cols:
        out.append(f"新增列：{added_cols} 个")
    if kept:
        out.append(f"已有列保持不变：{kept} 个")
    if new_terms:
        out.append(f"新增口径：{' / '.join(new_terms)}")
    return out or ["与已有词典一致，没有新增内容"]
