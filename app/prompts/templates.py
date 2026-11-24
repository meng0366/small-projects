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
6. 只输出代码，不要任何解释文字，不要 markdown 代码块标记。
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
