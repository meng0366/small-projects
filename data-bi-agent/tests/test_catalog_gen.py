"""词典描述自动生成（P1）的离线测试：全部不调用 LLM。

覆盖四层：
1. 类型映射与枚举识别：这两条决定"哪些事实会进词典"，误认的代价比漏认大得多；
2. 实测画像：真临时库上验外键（声明 / 推断）、日期识别、枚举取值；
3. 模型输出的解析与校验：脏格式能取出、不存在的表列必须丢掉；
4. 渲染：**必须能被 `yaml.safe_load` 原样读回** —— 格式错一次整个功能就废了。
"""
import os
import sqlite3
import tempfile
import unittest

import yaml

from app.tools.catalog_gen import (
    CatalogError,
    MAX_PROFILE_CHARS,
    build_draft,
    clean_hints,
    count_todos,
    describe_changes,
    format_profile,
    generate_descriptions,
    infer_type_from_values,
    is_enum_candidate,
    looks_like_dates,
    map_type,
    parse_draft,
    profile_schema,
    render_catalog_yaml,
    summarize_profile,
    term_source_exists,
    validate_draft,
    yaml_scalar,
)

DDL = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    region TEXT,
    customer_level TEXT
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    order_date TEXT NOT NULL,
    customer_id TEXT REFERENCES customers(customer_id),
    quantity INTEGER,
    discount REAL,
    amount REAL,
    channel TEXT
);
"""

CUSTOMERS = [("C001", "上海科技有限公司", "华东", "战略客户"),
             ("C002", "广州贸易有限公司", "华南", "重点客户"),
             ("C003", "北京信息技术有限公司", "华北", "普通客户"),
             ("C004", "成都电子商务有限公司", "西南", "重点客户")]
# 12 行而不是 4 行：表太小时"去重数 / 行数"天然接近 1，
# 4 个渠道在 5 行里占比 0.8，会正好卡在"近似唯一"的判据上 —— 那是样本问题不是规则问题
ORDERS = [("O00001", "2025-01-05", "C001", 2, 1.0, 1998.0, "Online"),
          ("O00002", "2025-02-11", "C002", 1, 0.95, 949.05, "JD"),
          ("O00003", "2025-03-20", "C001", 3, 0.9, 2697.3, "Online"),
          ("O00004", "2025-04-02", "C003", 1, 1.0, 899.0, "Tmall"),
          ("O00005", "2025-05-18", "C004", 2, 0.95, 1898.1, "Offline"),
          ("O00006", "2025-06-07", "C001", 1, 1.0, 999.0, "Online"),
          ("O00007", "2025-07-15", "C002", 2, 0.9, 1798.2, "JD"),
          ("O00008", "2025-08-23", "C003", 1, 1.0, 899.0, "Tmall"),
          ("O00009", "2025-09-09", "C004", 4, 0.95, 3796.2, "Offline"),
          ("O00010", "2025-10-30", "C001", 2, 1.0, 1998.0, "Online"),
          ("O00011", "2025-11-12", "C002", 1, 1.0, 999.0, "JD"),
          ("O00012", "2025-12-25", "C003", 3, 0.9, 2697.3, "Tmall")]


def make_db(ddl=DDL, with_fk=True) -> str:
    """建一个临时库并返回路径。`with_fk=False` 时去掉外键声明，用来测推断。"""
    if not with_fk:
        ddl = ddl.replace(" REFERENCES customers(customer_id)", "")
    tmp = tempfile.mkdtemp(prefix="catgen_")
    path = os.path.join(tmp, "demo.db")
    con = sqlite3.connect(path)
    con.executescript(ddl)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", ORDERS)
    con.commit()
    con.close()
    return path


LLM_DRAFT = {
    "tables": [
        {"name": "customers", "label": "客户表",
         "description": "客户主数据，含所在区域与等级。",
         "columns": {
             "customer_id": {"label": "客户编号"},
             "customer_name": {"label": "客户名称"},
             "region": {"label": "所在区域", "hints": ["地区", "区域"]},
             "customer_level": {"label": "客户等级", "hints": ["等级"]},
         }},
        {"name": "orders", "label": "订单表",
         "description": "每一条记录是一笔销售订单。",
         "columns": {
             "order_id": {"label": "订单编号"},
             "order_date": {"label": "下单日期"},
             "customer_id": {"label": "客户编号"},
             "quantity": {"label": "购买数量"},
             "discount": {"label": "折扣", "note": "1.0 表示无折扣"},
             "amount": {"label": "销售金额", "note": "单位元"},
             "channel": {"label": "渠道"},
         }},
    ],
    "terms": [
        {"name": "销售额", "definition": "orders.amount 求和", "source": "orders.amount"},
        {"name": "订单量", "definition": "orders.order_id 计数", "source": "orders.order_id"},
        {"name": "虚构指标", "definition": "orders.不存在的列 求和", "source": "orders.ghost"},
    ],
}


# ---------------------------------------------------------------- 类型与枚举

class TestMapType(unittest.TestCase):
    def test_declared_types(self):
        self.assertEqual(map_type("INTEGER"), "int")
        self.assertEqual(map_type("BIGINT"), "int")
        self.assertEqual(map_type("REAL"), "float")
        self.assertEqual(map_type("DOUBLE PRECISION"), "float")
        self.assertEqual(map_type("NUMERIC(10,2)"), "float")
        self.assertEqual(map_type("BOOLEAN"), "bool")
        self.assertEqual(map_type("DATETIME"), "date")
        self.assertEqual(map_type("VARCHAR(20)"), "string")

    def test_text_dates_are_dates(self):
        # SQLite 里日期普遍存成 TEXT，只看声明类型会全当成字符串，
        # 那样 collect_data_profile 就不会算时间覆盖范围
        self.assertEqual(map_type("TEXT", ["2025-01-05", "2025-02-11"]), "date")
        self.assertEqual(map_type("TEXT", ["2025-01-05 10:30:00"]), "date")
        self.assertEqual(map_type("TEXT", ["华东", "华南"]), "string")
        self.assertEqual(map_type("TEXT", ["2025-01-05", "华东"]), "string")

    def test_no_declared_type(self):
        # CREATE TABLE ... AS SELECT 出来的表常常没有声明类型
        self.assertEqual(map_type("", ["1", "2", "3"]), "int")
        self.assertEqual(map_type("", ["1.5", "2"]), "float")
        self.assertEqual(map_type("", ["2025-01-05"]), "date")
        self.assertEqual(map_type("", ["a", "b"]), "string")
        self.assertEqual(map_type("", []), "string")

    def test_helpers(self):
        self.assertTrue(looks_like_dates(["2025-01-05"]))
        self.assertFalse(looks_like_dates([]))
        self.assertFalse(looks_like_dates(["2025/01/05"]))
        self.assertEqual(infer_type_from_values(["1", "2"]), "int")
        self.assertEqual(infer_type_from_values(["1", "x"]), "string")


class TestEnumCandidate(unittest.TestCase):
    def test_text_enum(self):
        self.assertTrue(is_enum_candidate("region", "string", 5, 100))
        self.assertTrue(is_enum_candidate("channel", "string", 4, 12))
        # 表只有 5 行、4 个渠道时比例到 0.8，会被当成"近似唯一"排掉 ——
        # 这是刻意的取舍：样本太小时无法区分"分类"和"标识"，宁可漏认
        self.assertFalse(is_enum_candidate("channel", "string", 4, 5))

    def test_identifier_like_name_is_never_enum(self):
        # 回归：orders.customer_id 3877 行里 20 个不同值、是文本列，
        # 按"去重少 + 文本"会被认成枚举，然后 schema 里出现
        # 「customer_id 必须严格使用以下字面量：C001 / C002 / ...」
        for name in ("customer_id", "product_id", "id", "order_no",
                     "order_code", "user_uuid", "item_key"):
            self.assertFalse(is_enum_candidate(name, "string", 20, 3877), name)

    def test_every_row_different_is_not_enum(self):
        self.assertFalse(is_enum_candidate("customer_name", "string", 17, 20))

    def test_high_cardinality(self):
        self.assertFalse(is_enum_candidate("order_id", "string", 3877, 3877))
        self.assertFalse(is_enum_candidate("amount", "float", 238, 3877))

    def test_numeric_needs_suggestive_name(self):
        self.assertFalse(is_enum_candidate("quantity", "int", 4, 3877))
        self.assertFalse(is_enum_candidate("discount", "float", 3, 3877))
        self.assertTrue(is_enum_candidate("order_status", "int", 3, 100))
        self.assertTrue(is_enum_candidate("会员等级", "int", 3, 100))

    def test_other_exclusions(self):
        self.assertFalse(is_enum_candidate("region", "string", 5, 100, is_key=True))
        self.assertFalse(is_enum_candidate("order_date", "date", 5, 100))
        self.assertFalse(is_enum_candidate("region", "string", 0, 0))
        self.assertFalse(is_enum_candidate("region", "string", 1, 100))
        self.assertFalse(is_enum_candidate("region", "string", 30, 100))


# ---------------------------------------------------------------- 实测画像

class TestProfileSchema(unittest.TestCase):
    def test_declared_fk(self):
        profile = profile_schema(make_db(with_fk=True))
        rels = {(r["from_table"], r["from_column"]): r for r in profile["relationships"]}
        self.assertIn(("orders", "customer_id"), rels)
        self.assertEqual(rels[("orders", "customer_id")]["source"], "declared")
        self.assertEqual(rels[("orders", "customer_id")]["to_table"], "customers")

    def test_guessed_fk(self):
        profile = profile_schema(make_db(with_fk=False))
        rels = {(r["from_table"], r["from_column"]): r for r in profile["relationships"]}
        self.assertIn(("orders", "customer_id"), rels)
        self.assertEqual(rels[("orders", "customer_id")]["source"], "guessed")
        self.assertTrue(any("推断" in n for n in profile["notes"]))

    def test_columns_and_facts(self):
        profile = profile_schema(make_db())
        tables = {t["name"]: t for t in profile["tables"]}
        self.assertIn("orders", tables)
        self.assertEqual(tables["orders"]["rows"], 12)

        cols = {c["name"]: c for c in tables["orders"]["columns"]}
        self.assertEqual(cols["order_date"]["type"], "date")     # TEXT 但值是日期
        self.assertEqual(cols["amount"]["type"], "float")
        self.assertEqual(cols["quantity"]["type"], "int")
        self.assertEqual(cols["customer_id"]["fk"], "customers.customer_id")
        self.assertIn("Online", cols["channel"]["values"])
        # customer_id 是标识符，不该有 values
        self.assertEqual(cols["customer_id"]["values"], [])
        self.assertTrue(cols["order_id"]["pk"])

    def test_catalog_side_effects(self):
        profile = profile_schema(make_db())
        text = format_profile(profile)
        self.assertIn("全部取值", text)
        self.assertIn("主键", text)
        self.assertIn("表 customers", text)
        self.assertIn("2 张表", summarize_profile(profile))

    def test_missing_db(self):
        profile = profile_schema("/no/such/file.db")
        self.assertEqual(profile["tables"], [])
        self.assertTrue(profile["notes"])

    def test_not_a_database(self):
        tmp = tempfile.mkdtemp(prefix="catgen_")
        path = os.path.join(tmp, "fake.db")
        with open(path, "w", encoding="utf-8") as f:
            f.write("这不是数据库")
        profile = profile_schema(path)
        self.assertEqual(profile["tables"], [])
        self.assertTrue(any("读取库失败" in n for n in profile["notes"]))

    def test_empty_db(self):
        tmp = tempfile.mkdtemp(prefix="catgen_")
        path = os.path.join(tmp, "empty.db")
        sqlite3.connect(path).close()
        profile = profile_schema(path)
        self.assertEqual(profile["tables"], [])
        self.assertIn("没有用户表", " ".join(profile["notes"]))


# ---------------------------------------------------------------- 解析与校验

class TestParseDraft(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_draft('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(parse_draft('```json\n{"a": 1}\n```'), {"a": 1})

    def test_with_explanation_around(self):
        text = '好的，这是词典：\n{"a": 1}\n希望有帮助。'
        self.assertEqual(parse_draft(text), {"a": 1})

    def test_bad_input(self):
        self.assertIsNone(parse_draft(""))
        self.assertIsNone(parse_draft("没有 JSON"))
        self.assertIsNone(parse_draft("{这不是 JSON}"))
        self.assertIsNone(parse_draft("[1, 2]"))     # 顶层不是对象


class TestValidateDraft(unittest.TestCase):
    def setUp(self):
        self.profile = profile_schema(make_db())

    def test_clean_draft_has_no_error(self):
        issues = validate_draft(LLM_DRAFT, self.profile)
        # 假模型给的表列都真实存在，但有编造的口径
        errors = [i for i in issues if i["level"] == "error"]
        self.assertEqual(errors, [])
        self.assertTrue(any(i["kind"] == "bad_term" for i in issues))

    def test_unknown_table_and_column(self):
        draft = {"tables": [
            {"name": "ghost", "columns": {}},
            {"name": "orders", "columns": {"ghost_col": {"label": "幽灵"}}},
        ]}
        issues = validate_draft(draft, self.profile)
        kinds = [i["kind"] for i in issues if i["level"] == "error"]
        self.assertIn("unknown_table", kinds)
        self.assertIn("unknown_column", kinds)

    def test_missing_column_is_warn(self):
        draft = {"tables": [{"name": "orders", "columns": {"amount": {"label": "销售额"}}}]}
        issues = validate_draft(draft, self.profile)
        missing = [i for i in issues if i["kind"] == "missing_column"]
        self.assertGreater(len(missing), 1)
        self.assertTrue(all(i["level"] == "warn" for i in missing))

    def test_term_source(self):
        tables = {"orders": {"amount", "order_id"}}
        self.assertTrue(term_source_exists("orders.amount", tables))
        self.assertTrue(term_source_exists("orders.amount/orders.order_id", tables))
        self.assertFalse(term_source_exists("orders.ghost", tables))
        self.assertFalse(term_source_exists("ghost.amount", tables))
        self.assertFalse(term_source_exists("amount", tables))
        self.assertFalse(term_source_exists("", tables))


# ---------------------------------------------------------------- 合并

class TestBuildDraft(unittest.TestCase):
    def setUp(self):
        self.profile = profile_schema(make_db())

    def test_facts_come_from_profile_not_model(self):
        draft = {"tables": [{"name": "orders", "columns": {
            "amount": {"label": "金额", "type": "整数", "values": ["编造的"]},
            "channel": {"label": "渠道", "values": ["假渠道"]},
        }}]}
        out = build_draft(self.profile, draft)
        orders = next(t for t in out["tables"] if t["name"] == "orders")["columns"]
        self.assertEqual(orders["amount"]["type"], "float")        # 实测覆盖模型
        self.assertNotIn("values", orders["amount"])               # 模型编的枚举被丢掉
        self.assertEqual(set(orders["channel"]["values"]),
                         {"Online", "JD", "Tmall", "Offline"})     # 实测取值

    def test_label_priority(self):
        existing = {"tables": [{"name": "orders", "columns": {"amount": {"label": "手写金额"}}}]}
        llm = {"tables": [{"name": "orders", "columns": {"amount": {"label": "模型金额"}}}]}
        out = build_draft(self.profile, llm, existing)
        orders = next(t for t in out["tables"] if t["name"] == "orders")["columns"]
        self.assertEqual(orders["amount"]["label"], "手写金额")

        out2 = build_draft(self.profile, llm)
        orders2 = next(t for t in out2["tables"] if t["name"] == "orders")["columns"]
        self.assertEqual(orders2["amount"]["label"], "模型金额")

    def test_example_priority(self):
        existing = {"tables": [{"name": "orders", "columns": {"amount": {"example": 7598.1}}}]}
        out = build_draft(self.profile, None, existing)
        orders = next(t for t in out["tables"] if t["name"] == "orders")["columns"]
        self.assertEqual(orders["amount"]["example"], 7598.1)

        out2 = build_draft(self.profile, None)
        orders2 = next(t for t in out2["tables"] if t["name"] == "orders")["columns"]
        self.assertIn(orders2["amount"]["example"], [1998.0, 949.05, 2697.3, 899.0, 1898.1])

    def test_terms_filtered_and_kept(self):
        out = build_draft(self.profile, LLM_DRAFT)
        self.assertIn("销售额", out["terms"])
        self.assertIn("订单量", out["terms"])
        self.assertNotIn("虚构指标", out["terms"])     # source 指到不存在的列

    def test_existing_terms_not_overwritten(self):
        existing = {"terms": {"销售额": "手写口径"}}
        out = build_draft(self.profile, LLM_DRAFT, existing)
        self.assertEqual(out["terms"]["销售额"], "手写口径")
        self.assertIn("订单量", out["terms"])           # 缺的补上

    def test_relationships(self):
        out = build_draft(self.profile, None)
        self.assertTrue(any("orders.customer_id -> customers.customer_id" in r
                            for r in out["relationships"]))
        existing = {"relationships": ["手写的关系"]}
        self.assertEqual(build_draft(self.profile, None, existing)["relationships"],
                         ["手写的关系"])

    def test_no_model_at_all(self):
        out = build_draft(self.profile)
        orders = next(t for t in out["tables"] if t["name"] == "orders")["columns"]
        self.assertNotIn("label", orders["amount"])     # 没有中文名，等 TODO
        self.assertEqual(orders["amount"]["type"], "float")
        self.assertGreater(count_todos(out), 0)
        self.assertIn("列", describe_changes(out)[0])

    def test_hints_cleaned_and_only_for_enums(self):
        llm = {"tables": [{"name": "orders", "columns": {
            "channel": {"label": "渠道",
                        "hints": ["渠道", "渠道", " 下单渠道 ", "", "x" * 20,
                                  "途径", "来源", "入口", "平台"]},
            "amount": {"label": "金额", "hints": ["金额", "金额数"]},   # 非枚举列
        }}]}
        out = build_draft(self.profile, llm)
        orders = next(t for t in out["tables"] if t["name"] == "orders")["columns"]
        self.assertEqual(orders["channel"]["hints"],
                         ["渠道", "下单渠道", "途径", "来源", "入口", "平台"])   # 去重/去空/限 6 个
        self.assertNotIn("hints", orders["amount"])                    # 非枚举列不给 hints

    def test_clean_hints_helper(self):
        self.assertEqual(clean_hints(["渠道", " 渠道 "]), ["渠道"])
        self.assertEqual(clean_hints(["类"]), ["类"])            # 单字词根要保留
        self.assertEqual(clean_hints([""]), [])
        self.assertEqual(len(clean_hints([f"h{i}" for i in range(10)])), 6)


# ---------------------------------------------------------------- 渲染

class TestYamlScalar(unittest.TestCase):
    def test_plain_values_stay_plain(self):
        self.assertEqual(yaml_scalar("订单表"), "订单表")
        self.assertEqual(yaml_scalar("1.0 表示无折扣"), "1.0 表示无折扣")
        self.assertEqual(yaml_scalar(3999), "3999")
        self.assertEqual(yaml_scalar(1.0), "1.0")
        self.assertEqual(yaml_scalar(True), "true")

    def test_ambiguous_values_get_quoted(self):
        for text in ("2025-01-05", "001", "1.50", "true", "null", "~", "-3", ""):
            rendered = yaml_scalar(text)
            self.assertTrue(rendered.startswith('"'), f"{text} → {rendered}")
            self.assertEqual(yaml.safe_load(rendered), text)

    def test_special_chars_get_quoted(self):
        for text in ("含税: 是", "# 备注", "[华东]", "a,b", "{x}", "带货 & 包邮",
                     "有\u0009制表符", "首尾有空格 ", " '单引号' "):
            rendered = yaml_scalar(text)
            self.assertEqual(yaml.safe_load(rendered), text, f"{text} → {rendered}")

    def test_list(self):
        self.assertEqual(yaml_scalar(["华东", "华南"]), "[华东, 华南]")
        self.assertEqual(yaml.safe_load(yaml_scalar(["a,b", "c"])), ["a,b", "c"])


class TestRenderYaml(unittest.TestCase):
    def setUp(self):
        self.profile = profile_schema(make_db())
        self.draft = build_draft(self.profile, LLM_DRAFT)

    def test_roundtrip_is_equivalent(self):
        text = render_catalog_yaml(self.draft, source="demo.db", generated_at="2025-01-01 00:00")
        loaded = yaml.safe_load(text)
        self.assertEqual(set(loaded), {"tables", "relationships", "terms"})
        self.assertEqual([t["name"] for t in loaded["tables"]],
                         [t["name"] for t in self.draft["tables"]])
        for want, got in zip(self.draft["tables"], loaded["tables"]):
            self.assertEqual(want["label"], got["label"])
            self.assertEqual(want["description"], got["description"])
            self.assertEqual(set(want["columns"]), set(got["columns"]))
            for col, meta in want["columns"].items():
                self.assertEqual(meta["type"], got["columns"][col]["type"], col)
                if "values" in meta:
                    self.assertEqual(meta["values"], got["columns"][col]["values"], col)
                if "example" in meta:
                    self.assertEqual(meta["example"], got["columns"][col]["example"], col)
        self.assertEqual(self.draft["terms"], loaded["terms"])

    def test_block_style_for_long_columns(self):
        draft = {"tables": [{"name": "t", "label": "表", "description": "说明。",
                             "columns": {"region": {
                                 "label": "所在区域", "type": "string",
                                 "example": "华东",
                                 "values": ["华东", "华南", "华北", "西南", "华中"],
                                 "hints": ["地区", "区域", "省份", "城市", "大区"],
                                 "note": "按客户注册地归属"}}}],
                 "relationships": [], "terms": {}}
        text = render_catalog_yaml(draft)
        self.assertIn("      region:\n", text)      # 展开成 block
        loaded = yaml.safe_load(text)
        self.assertEqual(loaded["tables"][0]["columns"]["region"]["values"],
                         ["华东", "华南", "华北", "西南", "华中"])

    def test_todo_markers(self):
        draft = build_draft(self.profile)      # 没有模型 → 全是 TODO
        text = render_catalog_yaml(draft)
        self.assertIn("TODO", text)
        self.assertIn('label: ""   # TODO', text)
        self.assertGreater(count_todos(draft), 0)

    def test_messy_strings_survive(self):
        draft = {"tables": [{"name": "t", "label": "表格: 主表", "description": "# 以 2025 为准",
                             "columns": {"a": {"label": "值: 含税", "type": "string",
                                               "note": "1.0 表示无折扣 & 包邮 [已确认]"}}}],
                 "relationships": ["t.a -> u.b (多对一)"],
                 "terms": {"销售额": "t.a 求和: 单位元"}}
        loaded = yaml.safe_load(render_catalog_yaml(draft))
        self.assertEqual(loaded["tables"][0]["label"], "表格: 主表")
        self.assertEqual(loaded["tables"][0]["description"], "# 以 2025 为准")
        self.assertEqual(loaded["tables"][0]["columns"]["a"]["note"],
                         "1.0 表示无折扣 & 包邮 [已确认]")
        self.assertEqual(loaded["terms"]["销售额"], "t.a 求和: 单位元")

    def test_empty_description_gets_todo(self):
        draft = {"tables": [{"name": "t", "label": "表", "columns": {}}],
                 "relationships": [], "terms": {}}
        text = render_catalog_yaml(draft)
        self.assertIn('description: ""   # TODO', text)
        self.assertEqual(yaml.safe_load(text)["tables"][0]["description"], "")


# ---------------------------------------------------------------- 模型调用接线

class TestGenerateDescriptions(unittest.TestCase):
    def test_invoke_injected(self):
        profile = profile_schema(make_db())
        captured = {}

        def fake_invoke(system, user):
            captured["system"], captured["user"] = system, user
            return '{"tables": [], "terms": []}'

        out = generate_descriptions(profile, invoke=fake_invoke)
        self.assertIn("tables", out)
        self.assertIn("业务语义词典", captured["system"])
        self.assertIn("orders", captured["user"])
        self.assertIn("2 张表", captured["user"])

    def test_profile_too_large(self):
        profile = {"db": "x.db", "tables": [{"name": "t", "rows": 1, "columns": [
            {"name": f"col{i}", "type": "string", "raw_type": "TEXT", "distinct": 1,
             "nulls": 0, "examples": ["值" * 20], "values": [], "pk": False,
             "fk": "", "range": ""} for i in range(2000)]}],
            "relationships": [], "notes": []}
        with self.assertRaises(CatalogError):
            generate_descriptions(profile, invoke=lambda s, u: "")
        self.assertGreater(MAX_PROFILE_CHARS, 0)


# ---------------------------------------------------------------- 端到端

class TestEndToEnd(unittest.TestCase):
    def test_draft_activates_enum_guard(self):
        """词典的第二个用途：让"用户说了个库里没有的渠道"能被拦住。

        这正是自动生成必须产出 hints 的原因 —— 没有 hints，`enum_guard`
        对新库的枚举列完全不生效（它靠 hints 定位"他是在说哪一列的取值"）。
        """
        from app.schema.loader import collect_data_profile
        from app.tools.enum_guard import build_clarify_answer, find_unknown_enum

        db = make_db()
        profile = profile_schema(db)
        llm = {"tables": [{"name": "orders", "columns": {
            "channel": {"label": "渠道", "hints": ["渠道", "下单渠道"]}}}]}
        draft = build_draft(profile, llm)

        # 草稿里 channel 带着实测取值，collect_data_profile 才会把它当枚举列
        data = collect_data_profile(db, catalog=draft)
        flag = find_unknown_enum("只看京东渠道的销售额是多少？", data["enums"])
        self.assertIsNotNone(flag)
        self.assertEqual(flag["column"], "channel")
        self.assertEqual(flag["candidate"], "京东")
        self.assertIn("Online", flag["values"])

        answer = build_clarify_answer(flag)
        self.assertIn("渠道", answer)
        self.assertIn("Online", answer)

        # 用户说的值本来就在库里 → 不该拦
        self.assertIsNone(find_unknown_enum("只看 JD 渠道的销售额", data["enums"]))

    def test_draft_feeds_existing_loader(self):
        """生成的草稿必须能被现有 schema 拼装直接用 —— 否则这个功能没有出口。"""
        from app.schema.loader import build_db_schema_text

        db = make_db()
        profile = profile_schema(db)
        draft = build_draft(profile, LLM_DRAFT)
        loaded = yaml.safe_load(render_catalog_yaml(draft))

        # 1) 结构上就是个合法 catalog
        self.assertEqual([t["name"] for t in loaded["tables"]], ["customers", "orders"])

        # 2) 喂给 schema 拼装：中文名与枚举取值都要出现
        schema = build_db_schema_text(db, catalog=loaded)
        self.assertIn("订单表", schema)
        self.assertIn("销售金额", schema)
        self.assertIn("华东", schema)
        self.assertIn("customers.customer_id", schema)

    def test_merge_into_existing_catalog(self):
        """--merge 的核心承诺：已有的非空内容一个都不许变。"""
        db = make_db()
        profile = profile_schema(db)
        existing = {
            "tables": [{"name": "orders", "label": "我手写的订单表",
                        "columns": {"amount": {"label": "我手写的金额", "note": "含税"}}}],
            "relationships": ["手写关系"],
            "terms": {"销售额": "手写口径"},
        }
        draft = build_draft(profile, LLM_DRAFT, existing)
        orders = next(t for t in draft["tables"] if t["name"] == "orders")
        self.assertEqual(orders["label"], "我手写的订单表")
        self.assertEqual(orders["columns"]["amount"]["label"], "我手写的金额")
        self.assertEqual(orders["columns"]["amount"]["note"], "含税")
        self.assertEqual(draft["terms"]["销售额"], "手写口径")
        self.assertEqual(draft["relationships"], ["手写关系"])
        # 同时补上了新的
        self.assertEqual(orders["columns"]["channel"]["label"], "渠道")
        self.assertIn("订单量", draft["terms"])


if __name__ == "__main__":
    unittest.main()
