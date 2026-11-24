"""生成模拟销售数据（SQLite + CSV），供 Agent 分析练习使用。

运行：python data/generate_mock_data.py
数据内容：2025 年全年订单，涉及 4 大区、5 品类、20 个客户、30 个商品。
"""
import os
import random
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sales.db")
CSV_PATH = os.path.join(BASE_DIR, "sales_orders.csv")

# ---------- 1. 商品表 ----------
categories = {
    "手机": ["星云 X1", "星云 X1 Pro", "星云 Note", "轻舟 A5"],
    "笔记本": ["轻舟 Book 14", "轻舟 Book 16", "轻舟 Air", "锐影 Pro"],
    "平板": ["星云 Pad", "星云 Pad Pro", "轻舟 Tab", "悦读 Pad"],
    "配件": ["无线耳机", "智能手表", "充电宝", "保护壳"],
    "智能家居": ["智能音箱", "智能门锁", "扫地机器人", "智能灯泡"],
}
products = []
pid = 1
for cat, names in categories.items():
    for name in names:
        base_price = {
            "手机": (1999, 5999),
            "笔记本": (3999, 9999),
            "平板": (1499, 4999),
            "配件": (99, 899),
            "智能家居": (199, 2999),
        }[cat]
        price = random.randrange(base_price[0], base_price[1] + 1, 50)
        products.append({"product_id": f"P{pid:03d}", "product_name": name,
                         "category": cat, "unit_price": price})
        pid += 1
df_products = pd.DataFrame(products)

# ---------- 2. 客户表 ----------
regions = ["华东", "华南", "华北", "西南", "华中"]
cities = {
    "华东": ["上海", "杭州", "南京", "苏州"],
    "华南": ["深圳", "广州", "东莞", "佛山"],
    "华北": ["北京", "天津", "石家庄", "青岛"],
    "西南": ["成都", "重庆", "昆明", "贵阳"],
    "华中": ["武汉", "长沙", "郑州", "南昌"],
}
levels = ["战略客户", "重点客户", "普通客户"]
customers = []
for i in range(1, 21):
    region = random.choice(regions)
    customers.append({
        "customer_id": f"C{i:03d}",
        "customer_name": f"{random.choice(cities[region])}{random.choice(['科技', '贸易', '实业', '商贸', '网络'])}有限公司",
        "region": region,
        "customer_level": random.choice(levels),
    })
df_customers = pd.DataFrame(customers)

# ---------- 3. 订单表 ----------
orders = []
oid = 1
start = date(2025, 1, 1)
for day_offset in range(365):
    d = start + timedelta(days=day_offset)
    # 节假日后半月销量略高，制造趋势便于分析
    n_orders = random.randint(6, 14) + (3 if d.month in (6, 11, 12) else 0)
    for _ in range(n_orders):
        customer = random.choice(customers)
        product = random.choice(products)
        qty = random.choices([1, 2, 3, 5], weights=[55, 25, 15, 5])[0]
        discount = random.choice([1.0, 1.0, 0.95, 0.9])
        amount = round(product["unit_price"] * qty * discount, 2)
        orders.append({
            "order_id": f"O{oid:05d}",
            "order_date": d,
            "customer_id": customer["customer_id"],
            "product_id": product["product_id"],
            "quantity": qty,
            "unit_price": product["unit_price"],
            "discount": discount,
            "amount": amount,
        })
        oid += 1
df_orders = pd.DataFrame(orders)

# ---------- 4. 写入 SQLite ----------
engine = create_engine(f"sqlite:///{DB_PATH}")
df_products.to_sql("products", engine, index=False, if_exists="replace")
df_customers.to_sql("customers", engine, index=False, if_exists="replace")
df_orders.to_sql("orders", engine, index=False, if_exists="replace")

# ---------- 5. 写单表 CSV（join 后的宽表，供阶段 1 使用）----------
wide = df_orders.merge(df_customers, on="customer_id").merge(
    df_products.rename(columns={"unit_price": "list_price"}), on="product_id"
)
wide = wide[["order_id", "order_date", "customer_name", "region", "customer_level",
             "product_name", "category", "quantity", "unit_price", "discount", "amount"]]
wide.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print(f"数据库已生成：{DB_PATH}")
print(f"CSV 已生成：{CSV_PATH}")
print(f"商品 {len(df_products)} 条 / 客户 {len(df_customers)} 条 / 订单 {len(df_orders)} 条")
print(f"销售额合计：{df_orders['amount'].sum():,.2f} 元")
