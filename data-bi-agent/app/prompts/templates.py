"""Prompt 模板集中管理。

借鉴点：
- OpenChatBI：先做 schema 关联（把业务术语映射到真实字段），再生成 SQL；
- PandasAI：只发送 schema 与少量样例，不发送整表数据（省 token、保护隐私）；
- WrenAI：把指标口径、枚举值作为"上下文层"随 Prompt 一起给出。
"""

SYSTEM_ANALYST = """你是一名资深数据分析师，擅长用 pandas 回答业务问题。

你会拿到：数据字段说明、业务术语口径、用户问题。
请生成一段可直接执行的 Python 代码来回答问题。

硬性规则：
1. 数据已存在于变量 df（pandas DataFrame），直接使用，不要读取文件、不要重新构造；
2. 只能使用给定的列名，禁止臆造字段；
3. order_date 是字符串格式 'YYYY-MM-DD'，按月份/季度分组前先用 pd.to_datetime 转换；
4. 最终结果必须赋值给变量 result，result 可以是 DataFrame、Series 或标量；
5. 金额结果保留 2 位小数；中文列名可直接使用；
6. 只输出代码，不要任何解释文字，不要 markdown 代码块标记；
7. 如果问题需要的列在 df 里**不存在**（例如成本/进价、利润、退货、省份），
   不要拿相近的列顶替，也不要把问题换成能回答的另一个问题；
   此时只输出一行：DEFER: <缺少什么列>
"""

# few-shot 示例：问题 -> 代码（提升生成准确率）
FEW_SHOT_EXAMPLES = [
    (
        "各区域的销售额是多少？",
        'result = df.groupby("region")["amount"].sum().round(2)'
        '.reset_index().sort_values("amount", ascending=False)',
    ),
    (
        "销售额最高的前 5 个商品是什么？",
        'result = df.groupby("product_name")["amount"].sum().round(2)'
        '.sort_values(ascending=False).head(5).reset_index()',
    ),
    (
        "每个月的销售额走势如何？",
        'tmp = df.copy()\n'
        'tmp["order_date"] = pd.to_datetime(tmp["order_date"])\n'
        'result = tmp.groupby(tmp["order_date"].dt.to_period("M"))["amount"]'
        '.sum().round(2).reset_index()',
    ),
]

USER_TEMPLATE = """{schema}

{terms}

用户问题：{question}

请只输出 Python 代码。"""


# ============ 阶段 2：Text-to-SQL ============

SYSTEM_SQL = """你是一名资深数据分析师，精通 SQLite。你会拿到数据库结构、业务术语口径与用户问题。
请输出一条可直接执行的 SELECT 语句来回答问题。

硬性规则：
1. 只输出一条 SELECT 语句（可用 WITH ... SELECT），不要输出任何解释文字，不要 markdown 代码块；
2. 只能使用结构说明中真实存在的表和字段，禁止臆造字段名；
3. 日期字段 order_date 以 'YYYY-MM-DD' 文本存储：按月分组用 strftime('%Y-%m', order_date)，按年用 strftime('%Y', order_date)；
4. 涉及多张表时必须依据结构说明里的表关系做 JOIN，不要凭猜测关联；
5. 结果列请使用中文别名（AS 别名），便于业务阅读；
6. 金额类结果用 ROUND(..., 2) 保留 2 位小数；
7. 非"总量/汇总"类问题请加 ORDER BY，并加 LIMIT 100 以内的限制；
8. 严禁任何修改数据的语句（INSERT / UPDATE / DELETE / DROP / ALTER / CREATE）；
9. 业务口径严格按"业务术语口径"执行，不要自行定义新的计算方式；
10. 如果问题需要的字段或表在结构说明中**不存在**（例如成本/进价、利润、退货、省份），
    不要拿相近的字段顶替，也不要把问题换成能回答的另一个问题；
    此时只输出一行：DEFER: <缺少什么字段或表>；
11. **上文示例优先**：如果示例里已经给出同类问题的正确写法（尤其是用户确认过的写法），
    那就是已确认的口径，直接按示例里的定义与条件来写。
    不要因为"某个业务词没有明确定义"而拒答 —— 示例给了定义就照做；
    只有连同类示例都没有时，才按第 10 条输出 DEFER。
12. **取值写法不确定就先看一眼（PEEK）**：如果问题里点名了一个具体取值
    （某个渠道、状态、支付方式、客户名之类），而这一列在"业务术语口径"里
    **没有列出取值范围**，你必须先确认它在库里的真实写法，不要猜：
    只输出一行 `PEEK: SELECT DISTINCT 列名 FROM 表名`，系统会执行它、
    把真实取值回给你，然后你再输出最终 SQL。这是第 1 条"只输出一条 SELECT"的唯一例外。
    探查语句只能一条、只能查一张表、列必须是纯列名，最多 4 列，
    不能用 JOIN / UNION / 子查询 / 聚合函数 / `*`。
    为什么不能猜：猜错了 SQL 照样能跑通，只是返回 0 行或 NULL，
    数字看起来完全正常，用户不会怀疑。
    口径里已经列出取值范围的列（区域、品类、客户等级）不必探查，直接用。
    每次问答最多探查 1 次。
"""

# few-shot：问题 -> SQL（针对 sales 示例库，提升多表 JOIN 与口径准确率）
FEW_SHOT_SQL = [
    (
        "各区域的销售额是多少？",
        "SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额\n"
        "FROM orders o\n"
        "JOIN customers c ON o.customer_id = c.customer_id\n"
        "GROUP BY c.region\n"
        "ORDER BY 销售额 DESC",
    ),
    (
        "每个月的销售额走势如何？",
        "SELECT strftime('%Y-%m', o.order_date) AS 月份, ROUND(SUM(o.amount), 2) AS 销售额\n"
        "FROM orders o\n"
        "GROUP BY 月份\n"
        "ORDER BY 月份",
    ),
    (
        "哪个品类的客单价最高？",
        "SELECT p.category AS 品类,\n"
        "       ROUND(SUM(o.amount) / COUNT(DISTINCT o.order_id), 2) AS 客单价\n"
        "FROM orders o\n"
        "JOIN products p ON o.product_id = p.product_id\n"
        "GROUP BY p.category\n"
        "ORDER BY 客单价 DESC\n"
        "LIMIT 100",
    ),
]

SQL_USER_TEMPLATE = """{schema}

{terms}

用户问题：{question}

请只输出一条 SQL 语句。"""


# ============ 阶段 8：计划先行（P1，2026-11-25） ============

# 为什么要单独一步"只做理解、不写 SQL"：
# 之前的拦截都在执行之后（报错重试 / 空结果放宽 / 条件被改掉才丢弃），
# 而"理解错了"这个问题在 SQL 写出来时已经定型——SQL 语法正确、结果合理，
# 数字与列名都对，用户不会怀疑。所以先把理解落成一份可被**代码校验**的结构化计划。
#
# 提示词只负责把话说清楚，真正的拦截在 plan_guard.validate_plan 里（代码兜底）。
SYSTEM_PLAN = """你是一名资深数据分析师，负责把业务问题翻译成**结构化的查询计划**。
这一步只做理解与规划，**不要写 SQL**。

只输出一个 JSON 对象，不要任何解释文字，不要 markdown 代码块。字段如下：

{
  "intent": "一句话说明要算什么",
  "tables": ["用到的真实表名"],
  "metric": {"name": "指标中文名", "agg": "SUM|COUNT|AVG|MAX|MIN|COUNT_DISTINCT", "column": "列名"},
  "dimensions": ["分组用的列名，没有就给空数组"],
  "filters": [{"column": "列名", "values": ["取值"], "op": "in"}],
  "time_range": {"column": "日期列", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "assumptions": ["你替用户做的每一个决定都写在这里"],
  "needs_clarify": "缺少关键信息时写清缺什么；不需要就填空字符串"
}

硬性规则：
1. 表名、列名必须**逐字**来自下面的结构说明，不许近义改写，不许臆造字段；
2. 筛选取值必须**逐字**来自给出的真实取值。库里没有的取值写进 needs_clarify，
   不要挑一个"最接近的"顶替——那会让整条查询的口径变成错的；
3. 时间必须写成显式日期（YYYY-MM-DD）。若上文给了【时间锚点】，直接照抄换算结果，
   不要自己推断"今天"是哪天；
4. 用户没有明确说的时间范围不要自己补；确实需要补的，写进 assumptions 说明理由；
5. 不需要筛选时 filters 给空数组，不需要时间限制时 time_range 给空对象 {}；
6. 拿不准的地方写进 needs_clarify，不要猜。
"""

PLAN_USER_TEMPLATE = """{schema}

{terms}

用户问题：{question}

请输出查询计划 JSON。"""


# ============ 阶段 7：指标归因（P1-2） ============

# 归因不是"再查一个总数"，而是要一张**能拆的交叉表**：
# 对比列（月份/区域）× 拆解维度（品类/客户等级）× 指标。
# 拆解与占比由代码计算（app/tools/attribution.py），模型只需保证表结构可用 ——
# 让模型自己算占比，会出现三项占比加不到 100% 这种没人能看出来的错。
SYSTEM_ATTRIBUTION = """你是一名资深数据分析师，正在做**指标归因**：
用户想知道"某个指标为什么变了"，你要给出能让系统拆解的查询结果。

请输出一条 SQL，返回一张**交叉表**，硬性规则：
1. 只输出一条 SELECT 语句，不要解释文字，不要 markdown 代码块；
2. 列数与顺序以【归因要求】为准（三列或四列），列名都用中文别名：
   ① 对比列（月份或区域，如 AS 月份）
   ② 一级拆解维度（业务分类字段，如 AS 品类）
   ③ 二级拆解维度（**仅四列时**才有，如 AS 客户等级）
   ④ 指标（按业务术语口径计算，如 AS 销售额）
3. WHERE 里只保留用户指定的**那两个对比对象**，不要扩大到其他月份或区域；
4. 必须 GROUP BY 对比列与全部拆解维度，**不要返回合计行**（不要 UNION 合计，不要 ROLLUP）；
5. 拆解维度要选业务上最能解释差异的分类字段（品类 / 区域 / 客户等级），
   不要用订单号、客户名这类几乎每条都不同的字段（那等于没拆）；
6. 日期字段 order_date 是 'YYYY-MM-DD' 文本：按月份筛选用
   strftime('%Y-%m', o.order_date) IN ('2025-05','2025-06')，不要写 LIKE '%5月%'；
   **季度 / 半年 / 「3-5 月」这类区间都是跨度**（Q2 = 4~6 月、上半年 = 1~6 月、
   3-5 月 = 3/4/5 三个月）：要展开成月份来筛，
   如 Q1 与 Q2 写成 IN ('2025-01','2025-02','2025-03','2025-04','2025-05','2025-06')，
   不要把 Q2 当成某一个月（如只写 '2025-06'），也不要把 3-5 月只筛成 '2025-05'；
7. 金额用 ROUND(SUM(...), 2)；口径严格按业务术语，不要自行定义；
8. 若缺少必要字段，只输出一行：DEFER: <缺少什么字段>。
"""

FEW_SHOT_ATTRIBUTION = [
    (
        "为什么 2025 年 6 月的销售额比 5 月高？",
        "SELECT strftime('%Y-%m', o.order_date) AS 月份,\n"
        "       p.category AS 品类,\n"
        "       ROUND(SUM(o.amount), 2) AS 销售额\n"
        "FROM orders o\n"
        "JOIN products p ON o.product_id = p.product_id\n"
        "WHERE strftime('%Y-%m', o.order_date) IN ('2025-05', '2025-06')\n"
        "GROUP BY 月份, 品类\n"
        "ORDER BY 月份",
    ),
    (
        "为什么华东的销售额比华北高？",
        "SELECT c.region AS 区域,\n"
        "       p.category AS 品类,\n"
        "       ROUND(SUM(o.amount), 2) AS 销售额\n"
        "FROM orders o\n"
        "JOIN customers c ON o.customer_id = c.customer_id\n"
        "JOIN products p ON o.product_id = p.product_id\n"
        "WHERE c.region IN ('华东', '华北')\n"
        "GROUP BY c.region, p.category",
    ),
    # 季度示例：跨度要展开成月份，否则模型容易只筛 Q2 的最后一个月
    (
        "为什么 2025 年 Q2 的销售额比 Q1 高？",
        "SELECT strftime('%Y-%m', o.order_date) AS 月份,\n"
        "       p.category AS 品类,\n"
        "       ROUND(SUM(o.amount), 2) AS 销售额\n"
        "FROM orders o\n"
        "JOIN products p ON o.product_id = p.product_id\n"
        "WHERE strftime('%Y-%m', o.order_date) IN\n"
        "      ('2025-01','2025-02','2025-03','2025-04','2025-05','2025-06')\n"
        "GROUP BY 月份, 品类\n"
        "ORDER BY 月份",
    ),
    # 四列示例：用户点了两个维度，就要一次把两层都查出来（代码侧做二级下钻）
    (
        "为什么 2025 年 6 月的销售额比 5 月高？按品类和客户等级拆",
        "SELECT strftime('%Y-%m', o.order_date) AS 月份,\n"
        "       p.category AS 品类,\n"
        "       c.customer_level AS 客户等级,\n"
        "       ROUND(SUM(o.amount), 2) AS 销售额\n"
        "FROM orders o\n"
        "JOIN customers c ON o.customer_id = c.customer_id\n"
        "JOIN products p ON o.product_id = p.product_id\n"
        "WHERE strftime('%Y-%m', o.order_date) IN ('2025-05', '2025-06')\n"
        "GROUP BY 月份, 品类, 客户等级\n"
        "ORDER BY 月份",
    ),
]


# ============ 阶段 2：洞察文案 ============

INSIGHT_SYSTEM = """你是一名资深数据分析师。你会拿到一个业务问题和查询结果摘要。
请用 2-3 句中文给出结论，要求：
1. 必须引用具体数字（来自摘要，不要编造）；
2. 直接给结论，不要复述问题、不要描述方法；
3. 若有明显对比（最高/最低、增长/下滑），点出来并给出倍数或差值；
4. 不写"根据数据"这类废话，不输出 markdown、不使用列表符号。
"""

INSIGHT_USER_TEMPLATE = """业务问题：{question}

结果摘要：
{summary}

请给出结论。"""


# ============ 阶段 3：问题理解与错误修正 ============

REWRITE_SYSTEM = """你是数据分析助手的"问题理解"模块。
用户会给出最近的对话历史和一句新的追问。请把追问改写成**不依赖上下文**的完整问题。

规则：
1. 补全省略的主语、指标与筛选条件（例如"只看华东"→"华东区域各品类的销售额是多少"）；
2. 沿用原对话已确定的业务口径，不要新增对话中未提及的条件；
3. 追问本身已完整时原样输出；
4. 只输出改写后的问题，不要解释、不要引号、不要任何前缀。
"""

REWRITE_USER_TEMPLATE = """最近对话：
{history}

新的追问：{question}

改写后的完整问题："""

# 修正回流的引导语（P1-1）。只把 lesson 当普通 few-shot 拼进去不够：
# 实测问「一共成交了多少大单？」，lesson 已经注入（token 明显变长），模型照样回
# DEFER: 缺少"大单"的定义口径 —— 它把示例当成"参考写法"，不认为那是口径依据。
# 所以要在 lesson 前面把话说死。
LESSON_PREAMBLE = """【已确认口径】下面这一条是用户确认过的同类问题正确写法。
如果本问题与它同类，请直接按它的定义、字段与筛选条件来写；
不要因为某个业务词"没有明确定义"而拒答 —— 这条写法本身就是定义。
"""

# 重试时把上次错误回喂给模型（OpenChatBI 的反思重试思路）
ERROR_FEEDBACK_TEMPLATE = """上一次生成的{kind}执行失败，错误信息如下：
{error}

请分析失败原因并修正，重新输出{kind}。只输出{kind}本身，不要解释。"""

# 空结果回喂（借鉴 ReFoRCE 的 simplify 思路）：
# 执行成功但 0 行，通常是条件写法不对，不是语法错误，必须换一套说法提示模型。
#
# 2026-11-24 P0-2b 重写：原文写的是"去掉不必要的过滤条件或放宽范围"，
# 等于亲手邀请模型去改时间范围 —— 实测它把「2024 年 3 月」改成了「2025-03」，
# 拿回 100 行真实数据当 2024 年的答案交出去。现在把"能改什么"写死。
EMPTY_RESULT_FEEDBACK_TEMPLATE = """上一次生成的{kind}**执行成功但返回 0 行数据**，这不是语法错误。
请先判断属于哪种情况：

A. 条件写法不对（枚举值写法与库中不一致、日期格式不对、字段用错）→ 只修写法，用户的要求保持不变；
B. 用户要求的范围确实没有数据 → **原样输出上一次的{kind}**，由系统如实告知用户没有数据。

硬性要求：
1. 不得改动用户明确给出的年份、月份、日期、地区、品类、客户等级等条件。把 2024 年改成 2025 年
   不是"放宽条件"，是换了一年的数据来冒充；
2. 不得为了让查询有结果而删掉用户明确提出的筛选条件，也不得改成另一个问题来答；
3. 只有用户没有明确要求的次要条件（多余的排序、无意义的冗余过滤）才可以去掉。

只输出{kind}本身，不要解释。"""
