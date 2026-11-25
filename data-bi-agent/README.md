# 数据分析 Agent（Text-to-Pandas / Text-to-SQL）
![离线测试](https://github.com/meng0366/small-projects/actions/workflows/tests.yml/badge.svg?branch=main)

用自然语言向数据提问，Agent 自动生成并执行分析代码，返回结果表格与洞察。

> 定位：**轻量级、可读、可改**的数据分析 Agent 教学/原型项目。不追求企业级功能，重在把 Text2SQL 的完整链路跑通并保持代码简洁。

## 功能现状

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 环境与配置 | ✅ |
| 1 | 单表问答（Text-to-Pandas，命令行） | ✅ |
| 2 | 多表 SQL + 图表 + 洞察 | ✅ |
| 3 | LangGraph 编排 + 错误自愈 + 多轮记忆 | ✅ |
| 4 | Streamlit 网页界面 | ✅ |
| 5 | 评测 / 安全加固 / 部署 | ✅ |
| 6 | 命令行增强（子命令 / 子命令输出格式 / 会话续聊 / 追问建议） | ✅ |
| 改进 | 空结果自愈 / 聚合空识别 / 取值澄清 / 拒答与范围守卫 / 相对时间锚点 / 修正回流（teach、lessons）/ 指标归因（为什么变了）/ 持续集成（GitHub Actions）/ 链路追踪（trace_id）/ 计划先行（先出计划再写 SQL）/ 两段式探查（先看一眼真实取值）/ 词典自动生成（接新库不用手抄 catalog）/ **MCP server（供别的 Agent 直接调用）** / **SQL 复用（同一个问题再问不调模型）** | ✅ |

## 环境要求

- Python **3.11+**（已在 3.13 验证）
- 一个 OpenAI 兼容的 LLM API Key（默认 DeepSeek，可换任意兼容平台）
- 无需数据库服务，默认使用本地 SQLite / CSV

## 快速开始

### 1. 获取代码并进入目录

本项目是 [small-projects](https://github.com/meng0366/small-projects) 仓库的子项目：

```bash
git clone https://github.com/meng0366/small-projects.git
cd small-projects/data-bi-agent
```

### 2. 创建并激活虚拟环境

**Windows（PowerShell / CMD）**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 需要与我们完全一致的版本时，可用 `pip install -r requirements.lock.txt`。

只想让别的程序（Claude Desktop / IDE / 其他 Agent）调用它时，再装 MCP 那部分依赖：

```bash
pip install -r requirements-mcp.txt      # 在 requirements.txt 之上加 mcp
```

> 它不在主依赖里，是因为 mcp 会拖着 cryptography、pywin32 一串东西。
> 只用命令行或网页的人不必装。

### 4. 配置 API Key

```bash
# macOS / Linux
cp .env.example .env
# Windows
copy .env.example .env
```

编辑 `.env`，填入你的 Key：

```ini
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

> 换其他平台（OpenAI / 通义 / 智谱等）只需改这三项，接口需兼容 OpenAI SDK。

### 5. 自检

```bash
python hello_llm.py
```

看到模型返回中文即可。

### 6. 示例数据（已随仓库附带）

仓库内已包含一份模拟销售数据，克隆后可直接使用，**无需额外操作**：

- `data/sales.db` —— SQLite，3 张关联表（orders / customers / products）
- `data/sales_orders.csv` —— join 后的宽表（3877 行 × 11 列）

如需重新生成（随机种子固定，结果一致）：

```bash
python data/generate_mock_data.py
```

### 7. 开始提问

```bash
# 交互模式（默认；输入 q 退出）
python -m app.cli

# 单次提问
python -m app.cli "各区域的销售额是多少？"

# 也可以用子命令写法（两者等价）
python -m app.cli ask "各区域的销售额是多少？"
```

> 请在**项目根目录**下执行以上命令。
> 图表输出到 `output/`，为独立 HTML，双击即可在浏览器查看。

### 8. 多轮追问

交互模式下会记住上下文，可以直接用省略说法：

```
> 各区域的销售额是多少？
（返回五个区域的销售额）

> 只看华东
[改写] 华东区域的销售额是多少？
（自动加上 WHERE c.region = '华东'）
```

会话会自动保存，退出后可以接着聊（见下一节）。

## 命令行进阶用法（子命令）

> 2026-11-24 新增。设计参考 natural-sql / nlsql / simonw-llm 三个开源 CLI 的做法，
> 详见 `docs/参考项目记录.md`（该文档不上传仓库）。

四个子命令：`ask`、`chat`、`schema`、`logs`（外加 `version`），都用 `python -m app.cli` 调用。

### ask：一次提问，可导出

```bash
# 默认表格输出（带路由、生成的 SQL、洞察、追问建议）
python -m app.cli ask "各品类的销售额是多少？"

# 结构化输出，脚本里直接用
python -m app.cli ask "各区域的销售额是多少？" -f json

# 导出到文件（按扩展名识别格式）
python -m app.cli ask "各品类的销售额是多少？" -o 品类销售额.xlsx
python -m app.cli ask "各区域的销售额是多少？" -o 区域.csv

# 管道：问题从标准输入读，结果直接进文件
echo "各区域的销售额是多少？" | python -m app.cli ask - -f csv > 区域.csv
```

输出格式：`-f table`（默认，给人看）/ `markdown`（贴文档）/ `csv`（喂 Excel、透视表）/ `json`（喂脚本）。
**只有 `table` 带过程信息**，其余三种的标准输出就是纯数据，提示信息走 stderr，所以重定向不会脏。

`json` 长这样（节选）：

```json
{
  "question": "各区域的销售额是多少？",
  "ok": true,
  "route": "sql",
  "statement": "SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额 ...",
  "columns": ["区域", "销售额"],
  "rows": 5,
  "data": [{"区域": "华南", "销售额": 5064245.4}],
  "insight": "华南销售额最高……",
  "suggestions": ["只看「华南」的情况如何？", "各品类的销售额是多少？", "每个月的销售额走势如何？"],
  "elapsed": 1.2
}
```

### chat：交互会话，可续聊

```bash
python -m app.cli chat                  # 新会话
python -m app.cli chat -c               # 接着上一次会话继续问
python -m app.cli chat --session 销售复盘  # 指定会话名（存在则续上）
python -m app.cli chat --list-sessions  # 列出本地会话
```

会话存在 `~/.data-agent/sessions/*.json`（可用环境变量 `AGENT_HOME` 改根目录），
只存问答文字、不存结果表。交互中还支持斜杠命令：

| 命令 | 作用 |
| --- | --- |
| `/help` | 帮助 |
| `/schema` | 查看数据源与表结构 |
| `/logs` | 最近 5 条运行记录 |
| `/clear` | 清空上下文，开始新会话 |
| `/quit` | 退出（也可直接输入 `q`） |

### schema：这个库里有什么

```bash
python -m app.cli schema          # 表、行数、字段中文释义、表关系、指标口径
python -m app.cli schema --json   # 机器可读
```

回答"这个库有哪些表、销售额怎么算"这类问题不用花模型调用。

### logs：跑过什么、慢在哪

```bash
python -m app.cli logs              # 最近 10 条
python -m app.cli logs -n 5         # 最近 5 条
python -m app.cli logs --failed     # 只看失败
python -m app.cli logs --json       # 机器可读
python -m app.cli logs --trace 240001  # 看某一次问答的各阶段耗时（完整号 / 前缀 / 尾号都行）
```

显示时间、成败、路由、重试次数、结果行数、耗时、token 用量、追踪号。日志在 `output/logs/agent.jsonl`。
`--linear` 的汇总行没有 token 统计（线性路线没挂回调），那一列显示 `—`。

### 一次问答的链路：慢在哪、卡在哪

`ask` 的输出末尾会打印追踪号，网页端在回答下方也能看到。用它还原这一次问答的完整过程：

```bash
python -m app.cli ask "为什么 6 月的销售额比 5 月高？"
# ... [追踪] 20261124-225725-240001（看这次各阶段耗时：python -m app.cli logs --trace 20261124-225725-240001）

python -m app.cli logs --trace 20261124-225725-240001
# 阶段          耗时  关键信息
# understand   33ms  route=sql
# generate    837ms  statement_len=158
# execute      13ms  rows=5
# answer      942ms
# 汇总：成功 · 5 行 · 2 次模型调用 · tokens 2854/33 · 1.83s
```

日志里两类行共用一个 `trace_id`：**阶段事件**（每个节点一行，带 `stage`）和末尾的
**汇总行**（没有 `stage`）。generate 慢说明改写或生成卡住，execute 慢说明 SQL 本身重；
链里出现两次 `generate → execute` 就是走了重试环。`--trace` 只给尾号 `240001` 也能查到。

复杂问题会在 understand 之后多出一个 `plan` 阶段（出现在链里就说明这次走了计划先行，
`--no-plan` 可跳过）；计划本身不触发时不会有这一行，汇总行里会记 `plan=跳过原因`。

只有状态机路线有 `understand` / `clarify` / `reflect` 这些阶段；`--linear` 路线同样有链，
但固定是 `generate → execute → answer`，SQL 失败回退 pandas 时会出现第二组
`generate → execute`（带 `fallback`），一眼能看出"回退过"：

```bash
python -m app.cli ask "各区域的销售额是多少？" --linear --no-chart
# ... [追踪] 20261124-232853-080001
# 阶段          耗时  关键信息
# generate    1125ms  route=sql · statement_len=163
# execute        5ms  route=sql · rows=5
# answer       969ms  rows=5 · chart=none
# 汇总：成功 · 5 行 · token 未统计（线性路线） · 2.1s（线性编排）
```

日志只记标量字段，**SQL 全文与结果表不会落盘**；这份字段白名单集中在
`app/logging_utils.py` 的 `trace_fields()`，状态机与线性编排共用同一份，不会各记一套。

### teach / lessons：答错一次，以后都对

模型不知道的业务口径（"大单指多少钱"、"活跃客户怎么算"），教一次就能长期生效：

```bash
# 教一条：问题 → 正确 SQL
python -m app.cli teach "一共成交了多少大单？" \
  --sql "SELECT COUNT(*) AS 大单数 FROM orders WHERE amount >= 10000" \
  -n "大单 = 单笔金额 10000 元及以上"

# 这类问题本来就该拒答，也能教（输出写成 DEFER:）
python -m app.cli teach "每个商品的毛利率是多少？" \
  --sql "DEFER: products 表只有标价 unit_price，没有成本/进价字段"

python -m app.cli lessons                      # 看已沉淀的记录
python -m app.cli lessons --match "大单有多少"   # 看这个问题会命中哪几条、相关度多少
python -m app.cli lessons --rm L001            # 删掉一条
python -m app.cli lessons --json                # 机器可读（脚本/巡检用）
```

参数速查：

| 命令 | 参数 | 作用 |
| --- | --- | --- |
| `teach` | `问题`（位置参数） | 答错的那个问题，**写原话**，匹配靠它 |
| | `-s, --sql` | 正确的 SQL（**必填**）；本来该拒答的写 `DEFER: 原因` |
| | `-n, --note` | 备注：为什么这么改（只在本地展示，不进提示词） |
| | `--force` | 跳过"入库前真跑一遍"的校验（不建议） |
| | `--db PATH` | 这条修正属于哪个库（按库指纹隔离） |
| `lessons` | `--match QUESTION` | 只显示会被这个问题命中的记录及相关度 |
| | `--rm ID` | 删掉指定记录（同时清空该库的 SQL 复用缓存） |
| | `--json` | 以 JSON 输出 |
| | `--db PATH` | 指定库 |

几条规则：

- **存在 `~/.data-agent/lessons/<库指纹>.jsonl`**，按库的表结构隔离 —— 换个库不会把旧库的写法带过去。
- **入库前会真的跑一遍**，跑不通的 SQL 拒绝入库（`--force` 可跳过，不建议）；只允许单条 SELECT。
- **只注入相关的**：优先按"库里没有的业务词"（如"大单"）匹配，其次才是去掉通用词后的字面相似；
  维度/取值/年份对不上（华南 vs 华东、区域 vs 品类、2024 vs 2025）一律不注入 —— 宁可少给一条示范，
  也不给一条条件对不上的。
- 教一次的效果：问「大单主要来自哪些品类」「华南的大单有多少笔」「大单的平均金额是多少」，
  三个不同的问题都会自动带上 `amount >= 10000`；问「今天成交了多少笔」则完全不受影响。

### catalog：接自己的库，自动生成词典

`catalog.yaml` 原来是手写的 —— 接一个新库得先搞清楚每张表每列什么意思，再照 YAML 格式抄一遍。
现在一条命令出草稿：

```bash
python -m app.cli catalog --db 我的库.db --dry-run   # 先看代码能自动认出什么（零成本，不调模型）
python -m app.cli catalog --db 我的库.db             # 生成 catalog.draft.yaml
python -m app.cli catalog --db 我的库.db --merge app/schema/catalog.yaml   # 只补空缺，不动已有内容
python -m app.cli catalog --db 我的库.db -o 我的草稿.yaml   # 换个输出路径
python -m app.cli catalog --db 我的库.db --json      # 只打到标准输出，不写文件（脚本用）
```

| 参数 | 作用 |
| --- | --- |
| `--db PATH` | 要分析的 SQLite 库（**必填**，只支持 SQLite） |
| `-o, --out PATH` | 草稿写到哪里，默认 `catalog.draft.yaml` |
| `--dry-run` | 只跑代码实测，不调模型 —— 零 token，先看自动识别到什么程度 |
| `--merge PATH` | 把草稿合并进已有词典：**已有的不动**，只补空缺 |
| `--json` | 结果以 JSON 打到标准输出，不落文件 |

分工是这样的，这也是它不会写错的原因：

| 谁 | 负责 |
| --- | --- |
| 代码 | 事实：类型、样例值、枚举取值、主外键、行数 |
| 模型 | 语义：中文名、一句话说明、提问线索词（`hints`）、指标口径建议 |
| 你 | 拍板：草稿里带 TODO，确认后覆盖 `app/schema/catalog.yaml` 才生效 |

生成出来大概这样：

```yaml
      channel:
        label: 销售渠道
        type: string
        example: Online
        values: [JD, Offline, Online, Tmall]
        hints: [渠道, 销售渠道, 下单渠道]
```

**为什么收益比看上去大**：词典不只是给模型看的，它还养着取值校验。同一个问题
「只看京东渠道的销售额是多少？」：

| 配置 | 结果 |
| --- | --- |
| 有这份词典 | 1 次调用直接澄清：「京东」不在销售渠道取值里，现有 JD / Offline / Online / Tmall，**不生成 SQL** |
| 没有词典 | 生成 `WHERE channel = '京东'` → 一行 NULL → "没有匹配到数据"（数字看着正常，其实是写法错了） |

不想覆盖内置词典，就把它指到别处：

```bash
AGENT_CATALOG=我的词典.yaml python -m app.cli ask "各渠道的销售额是多少？" --db 我的库.db
```

两点注意：**只支持 SQLite**；列的中文名由模型提议，**口径名要自己过一眼** ——
模型写「销售金额」而你同事都问「销售额」，这个口径就命中不了（草稿注释里也写了这句）。

### serve：作为 MCP server 被别的 Agent 调用

前面所有用法都是"人问、机器答"。但数据分析最常见的调用方其实是**另一个 Agent** ——
写周报的 Agent 想知道本月销售额，它不该自己写 SQL，也不该让人复制粘贴。
`serve` 把三件事暴露成 MCP 工具：

```bash
pip install -r requirements-mcp.txt
python -m app.cli serve                              # stdio，默认用 data/sales.db
python -m app.cli serve --db D:/data/我的库.db        # 指定默认库
```

在 Claude Desktop 的 `claude_desktop_config.json` 里加一段（路径换成你自己的）：

```json
{
  "mcpServers": {
    "data-bi-agent": {
      "command": "D:/agent/small-projects/data-bi-agent/.venv/Scripts/python.exe",
      "args": ["-m", "app.cli", "serve", "--db", "D:/agent/small-projects/data-bi-agent/data/sales.db"],
      "cwd": "D:/agent/small-projects/data-bi-agent"
    }
  }
}
```

`cwd` 别省 —— `-m app.cli` 要靠它找到包，否则客户端只会报"服务器启动失败"。

| 工具 | 干什么 | 对应命令 |
| --- | --- | --- |
| `ask` | 用中文问一个数据问题，返回结果数据 + 生成的 SQL + 执行信息 | `ask -f json` |
| `get_schema` | 看有哪些表/列（含中文名）、指标口径、表关系 | `schema --json` |
| `generate_catalog` | 给陌生的库生成词典草稿（**会写文件**，需人工确认） | `catalog` |

`ask` 的参数：`question`（必填）、`db`、`plan` / `peek`（默认开，两项自检）、
`insight` / `chart`（默认关，一个多花 token 一个写图片文件）、`linear`、
`max_rows`（默认 200）。

返回长这样（截断处能看到 `truncated`）：

```json
{
  "ok": true,
  "route": "sql",
  "statement": "SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额 ...",
  "columns": ["区域", "销售额"],
  "rows": 5,
  "truncated": false,
  "data": [{"区域": "华东", "销售额": 2345678.9}],
  "plan": {"used": true, "content": {"metric": "SUM(amount)", "dimensions": ["region"]}},
  "trace_id": "20261126-002002-900001"
}
```

三个刻意的设计，都是踩过才想清楚的：

| 选择 | 为什么 |
| --- | --- |
| **stdio 下一行 print 都没有** | stdout 是 JSON-RPC 通道，混一行"正在启动"客户端就连不上。所以工具只返回字符串，不带任何业务打印（CLI 那套 `run_*` 一个都不能直接调） |
| **数据默认截断到 200 行，并写明 `truncated` / `rows`** | 返回值进的是另一个模型的上下文。悄悄只给一部分，比明说截断危险得多 |
| **出错返回 `{"ok": false, "error": "..."}` 而不是抛异常** | 对面看到一屏 traceback 只会说"工具坏了"；看到结构化错误才会换个问法，或者去告诉用户原因 |

只读：`ask` 走的是同一套只读库 + SQL 安全校验，不会改你的数据。
唯一有写操作的是 `generate_catalog`，它只写 `catalog.draft.yaml`，绝不覆盖
`app/schema/catalog.yaml`；`dry_run=true` 时连模型都不调。

想关掉某次调用的自检省钱：`plan=false` / `peek=false`；
`--no-plan` / `--no-peek` 那两个命令行参数对 MCP 无效（工具参数优先）。

### 公共参数

| 参数 | 作用 | 适用 |
| --- | --- | --- |
| `--db PATH` | 换一个 SQLite 库文件（默认 `data/sales.db`）；路径不存在直接报错退出 | ask / chat / schema / catalog（必填） |
| `--no-chart` | 不生成图表 | ask / chat |
| `--no-insight` | 不生成洞察文案（省一次模型调用） | ask / chat |
| `--no-plan` | 不先出查询计划，直接生成 SQL（省一次模型调用，复现旧行为） | ask / chat |
| `--no-peek` | 不允许两段式探查（模型不确定取值时先看一眼真实数据） | ask / chat |
| `--no-cache` | 不复用历史 SQL，每次都让模型重写（调口径、怀疑数字时用） | ask / chat |
| `--linear` | 用线性编排（无重试、无多轮记忆），便于对比；同样有追踪链路 | ask / chat |
| `--pandas` | 仅线性模式：强制走单表 pandas | ask / chat |
| `-q, --quiet` | 不打印生成的 SQL / 代码 | ask |

**退出码**：`0` 成功 · `1` 分析失败 · `2` 参数或数据源错误，可直接用在 shell 脚本里：

```bash
if python -m app.cli ask "各区域销售额" -f json -q > out.json; then
  echo "成功"
else
  echo "失败，看 logs: python -m app.cli logs --failed"
fi
```

## 网页界面（Streamlit）

命令行之外还提供一个聊天式网页界面，适合演示或给非技术同事使用：

```bash
streamlit run app/main.py
```

浏览器会自动打开 <http://localhost:8501>（未自动打开时手动访问）。

界面能力：

| 区域 | 内容 |
| --- | --- |
| 左侧栏 | 数据源概览（表名 + 行数）、可展开查看随问题发给模型的 Schema、图表/洞察开关、**「先出查询计划」开关**、6 个示例问题、清空对话 |
| 对话区 | 一问一答，保留多轮上下文，支持「只看华东」这类追问 |
| 结果区 | 中文洞察结论、结果表格（可下载 CSV）、交互式图表（折线/柱状/饼图/散点）、可展开查看生成的 SQL 或 Python、**可展开查看查询计划与其中的假设**、**追问建议按钮**（点一下接着问） |
| 异常处理 | 分析失败时给出错误摘要与重试记录，页面不崩 |

> 网页与命令行共用同一套状态机，行为一致；界面侧只负责渲染，不重复业务逻辑。

## 示例问题

单表（pandas 路径）与多表（SQL 路径）都能回答：

- 各区域的销售额是多少？
- 销售额最高的前 5 个商品是什么？
- 每个月的销售额走势如何？
- 哪个品类的客单价最高？
- 哪个客户等级的销售额贡献最大？
- 2025 年总销售额是多少？
- 为什么 6 月的销售额比 5 月高？（归因：按品类拆出各分项贡献）
- 为什么华东的销售额比华北高？（归因：维度对比也能拆）
- 为什么 Q2 的销售额比 Q1 高？（归因：季度/半年这种跨度也能识别）
- 为什么 3-5 月的销售额比 1-2 月高？（归因：自己写的区间同样按跨度处理）

## 项目结构

```
app/
├── config.py            # 配置与密钥读取（环境变量 / .env / st.secrets）+ 安全限额
├── llm.py               # LLM 客户端与代码块清洗
├── context.py           # 运行时上下文（数据/Schema/术语，惰性加载）
├── state.py             # AgentState 状态结构
├── graph.py             # LangGraph 状态机（含重试环）+ 运行日志
├── logging_utils.py     # 运行日志、token 计数与 trace_id 链路追踪（阶段事件 + 汇总行）
├── agent.py             # 线性编排（--linear 时使用，也写阶段日志）
├── cli.py               # 命令行入口（子命令 ask / chat / schema / logs / teach / lessons / catalog / serve / version）
├── payload.py           # 结构化输出：CLI 的 -f json 与 MCP 工具共用同一份 payload
├── mcp_server.py        # MCP server：把 ask / get_schema / generate_catalog 暴露给别的 Agent
├── render.py            # 结果渲染与导出（table / markdown / csv / json / xlsx）
├── sessions.py          # 会话持久化（~/.data-agent/sessions，支持 chat -c 续聊）
├── main.py              # Streamlit 网页界面（聊天式问答）
├── schema/
│   ├── loader.py        # 数据加载 / Schema 描述（DataFrame 版 + 数据库版）
│   └── catalog.yaml     # 业务语义词典（字段含义、指标口径、枚举值）
├── prompts/templates.py # Prompt 模板（代码/SQL/洞察/问题改写/错误修正）
├── nodes/
│   ├── cache.py         # SQL 复用：命中历史查询就跳过生成与计划（首个节点）
│   ├── understand.py    # 问题理解：多轮改写 + 路由判定 + 取值校验
│   ├── generate.py      # Text-to-Pandas / Text-to-SQL 生成
│   ├── execute.py       # Python 代码安全执行（AST 校验 + 超时 + 截断）
│   ├── reflect.py       # 反思重试与换路、空结果判定
│   ├── plan.py          # 计划先行：复杂问题先出计划，交给 plan_guard 校验
│   └── answer.py        # 结果压缩 + 洞察生成
└── tools/
    ├── sql_tool.py      # SQL 安全校验与只读执行（含查询超时）
    ├── chart_tool.py    # 规则式自动选图（Plotly）
    ├── suggest_tool.py  # 追问建议（规则式，零 token）
    ├── enum_guard.py    # 取值校验：库里没有的值先澄清，不硬映射（规则式，零 token）
    ├── scope_guard.py   # 范围守卫：重试不许改用户的年份/季度/半年/月份/取值（防止拿别的范围冒充）
    ├── time_anchor.py   # 相对时间锚点：系统给定"今天"，把上个月/最近 N 天换算成显式区间
    ├── lesson_store.py  # 修正回流：把"问题 → 正确 SQL"沉淀成 few-shot（按库指纹隔离）
    ├── attribution.py   # 指标归因：识别"谁跟谁比"、把总变化拆成分项贡献（纯代码算）
    ├── plan_guard.py    # 计划校验：判定何时需要计划、解析与核对表/列/时间/取值（纯规则，零 token）
    ├── peek.py          # 两段式探查：白名单校验 + 只读执行 + 把取值压成可回灌的清单
    ├── catalog_gen.py   # 词典自动生成：实测画像 + 校验模型返回 + 合并三方 + 渲染带注释的 YAML
    ├── sql_cache.py     # SQL 复用：缓存键、七条不复用判据、结果指纹、LRU 淘汰（纯函数层）
    └── pandas_tool.py   # 预留占位（当前留空；pandas 路线实际在 nodes/execute.py）
data/
├── sales.db             # 模拟销售库（SQLite，3 张关联表）
├── sales_orders.csv     # 模拟销售宽表
└── generate_mock_data.py # 模拟数据生成脚本
evals/
├── cases.yaml           # 53 条评测用例（43 条有金标准 + 6 条该拒答 + 4 条该澄清）
├── judge.py             # 判分器（数值容差、列名/列顺序无关、refuse/clarify 模式）
├── run_eval.py          # 跑评测并产出报告
└── reports/             # 评测报告（仅 baseline 入库，临时报告已忽略）
tests/
├── test_smoke.py        # 离线冒烟测试（34 个用例）
├── test_graph.py        # 状态机与重试测试（12 个用例，全 mock LLM）
├── test_guard.py        # 取值校验与澄清分支测试（7 个用例）
├── test_scope.py        # 范围守卫测试（40 个用例，含"改年份换出数据"的拒绝、区间锚点、季度与半年多写法）
├── test_time_anchor.py  # 相对时间锚点（26 个用例：15 类相对时间含半年、覆盖判定、反例、两引擎）
├── test_lesson.py       # 修正回流测试（27 个用例：入库校验、指纹隔离、相关度正反例、注入位置）
├── test_attribution.py  # 指标归因测试（60 个用例：语序定基期、跨年歧义、合计行、比率指标、降级、二级拆解、季度/半年/自定义区间）
├── test_trace.py        # 链路追踪测试（36 个用例：追踪号唯一性、三级匹配、重试链、字段白名单、线性链）
├── test_plan.py         # 计划先行测试（42 个用例：触发判定、JSON 解析、表/列/时间/取值/半年校验、两引擎）
├── test_peek.py         # 两段式探查测试（55 个用例：前缀解析、白名单正反例、回灌格式、额度、两引擎）
├── test_catalog_gen.py  # 词典自动生成测试（48 个用例：类型映射、枚举判据、外键推断、YAML 往返、激活 enum_guard）
├── test_mcp_server.py   # MCP 测试（33 个用例：工具注册与参数 schema、截断标注、错误结构化、stdout 洁净、未装 mcp 的提示）
├── test_prompt_budget.py # 提示词预算测试（15 个用例：schema 只注入一次、模板不许写回 {schema}、新写法必须省 40% 以上）
├── test_sql_cache.py    # SQL 复用测试（32 个用例：键与隔离、七条不复用规则、结果指纹、过时 SQL 作废、两引擎）
├── _env.py              # 测试环境隔离（临时 AGENT_HOME、默认关闭 SQL 复用，防止用例互相污染）
├── test_ui.py           # 网页界面测试（8 个用例，全 mock LLM）
├── test_eval.py         # 评测集与判分测试（34 个用例，含 refuse/clarify）
├── test_security.py     # 安全防护测试（16 个用例）
├── test_deploy.py       # 部署配置测试（12 个用例，含配置总闸）
└── test_cli.py          # 命令行与新模块测试（55 个用例，全离线）
CHANGELOG.md             # 变更说明（面向协作方：改了什么、行为差异、同步清单）
LICENSE                  # MIT（版权人 雷佳山）
.env.example             # 密钥模板（复制成 .env 再填，.env 本身永不上传）
.dockerignore            # 构建镜像时排除 .env / 测试 / 本地文档
requirements.txt         # 运行依赖
requirements.lock.txt    # 锁版本依赖（要与我们完全一致的环境时用）
requirements-mcp.txt     # 可选依赖：只有在 `serve` 跑 MCP server 时才需要
hello_llm.py             # 自检脚本：确认 Key 与网络通不通
Dockerfile               # 单镜像部署（含中文字体与健康检查）
.github/workflows/tests.yml  # CI：每次推送跑离线测试并构建镜像（在仓库根，与项目同级）
.streamlit/config.toml   # 界面主题与服务器配置
```

## 工作流程

```
问题 + 历史对话
 ↓
understand：有历史则改写成完整问题；判定走 SQL 还是 pandas
           并做取值校验——提到的地区/品类/客户等级库里没有时，先问清再动手
 ↓
  ├─ 取值对不上 ─→ clarify：说明"库里没有这个值"，列出真实可选值（零 token，不生成）
  ↓
plan：复杂问题先出一份结构化计划（指标 / 表 / 维度 / 过滤 / 时间 / 假设），
      交给代码校验——表名、列名、时间是否真有数据、取值是否在库里
  ↓
  ├─ 计划发现时间没数据 ─→ clarify：说明数据只覆盖到哪，不平移到别的时段
  ├─ 计划发现取值不存在 → clarify：列出真实可选值
  └─ 计划说库里没这个字段 → answer：如实说明缺什么
  ↓
generate：生成 SQL 或 pandas 代码（重试时带上一次的真实报错）
          先挑出已沉淀的修正记录：只注入与本问题相关的（教过的口径、该拒答的示范）
          库里缺字段/表时输出 `DEFER: 原因`，不猜测、不换成别的问题
 ↓
  └─ 取值没把握 ─→ PEEK：只读看一眼真实取值，回灌后重写（每次问答最多一次）
                   如 `PEEK: SELECT DISTINCT channel FROM orders` → Online / Offline / JD / Tmall
 ↓
execute：SQL 走只读库 / 代码走受限命名空间
          两道守卫：`DEFER` 不当代码执行；空结果重试不许改动用户给定的年份/月份/取值
 ↓
  ├─ 成功且有数据 ─→ answer：规则选图（Plotly HTML）+ 统计分析 → 中文洞察
  ├─ 声明答不了 ──→ answer：说明缺哪个字段/数据，并列出库里真实覆盖的范围
  ├─ 聚合结果为空 ─→ answer：不重试（SUM 返回 NULL 说明条件内确实没数据，
  │                  放宽会把别的时段/地区的数算进同一指标），直接给事实
  ├─ 成功但 0 行 ─→ reflect：不换路，只提示"修写法、别改用户条件"重写一次
  │                   → 仍为空则如实说明"没有匹配数据"，不空转、不编结论
  └─ 失败 ────────→ reflect：记录错误、SQL 失败则换 pandas → 回到 generate（最多 2 次）
```

**为什么要先出一份计划**：直接让模型写 SQL，它会把"客单价"静默算成 `AVG(amount)`、
把没指定时间的问题偷偷限成"数据里最后 30 天"，而这些都藏在一条能跑通的 SQL 里，
看不出错。计划先行把意图先摊开成可检查的结构：

```
只看办公用品的销售额是多少？
→ 计划：指标 = SUM(amount)；表 = orders, products；维度 = 无；过滤 = products.category = '办公用品'
→ 代码校验：category 的取值里没有「办公用品」→ 拦下，不生成 SQL
→ 回答：库里 category 只有 手机 / 笔记本 / 平板 / 配件 / 智能家居，你要看哪一个？
```

计划里的**假设会直接展示给用户**（"客单价按 SUM(amount)/COUNT(order_id) 计算"、
"用户未指定时间范围，因此不限制日期"），不再是一个看不见的默认动作。
只有复杂问题才走这一步（含过滤语义、比率/派生指标、长问题），简单聚合直接生成——
这样每条计划约 2.5k 输入 token 的开销只花在真正会出错的提问上（评测集触发率 23%）。
嫌慢或想复现旧行为用 `--no-plan`，网页侧边栏也有对应开关。

**为什么要"先看一眼"**：`catalog.yaml` 里写了取值范围的列（区域、品类、客户等级）轮不到猜，
但真实库里还有大量没写进词典的列 —— 渠道、状态、支付方式、客户名。模型对这些列的取值只能猜，
而猜错**不会报错**：SQL 照样跑通，只是返回 0 行或 NULL，数字看起来完全正常。

```
只看京东渠道的销售额是多少？      # 库里 channel 的实际取值：Online / Offline / JD / Tmall
→ 猜：   WHERE channel = '京东'   → 一行 NULL → "当前条件下没有匹配到数据"
→ 探查： PEEK: SELECT DISTINCT channel FROM orders
         [探查] orders.channel → 4 个取值：Online / Offline / JD / Tmall
→ 改写： WHERE channel = 'JD'     → 4,563,724.05 ✅
```

探查是**只读**的，形态被收窄到"点名列的单表查询"（禁 JOIN / UNION / 子查询 / 聚合 / `*`，
最多 4 列、最多 50 行），每次问答最多一次，结果压成取值清单再回灌。
**它不产生答案** —— 看一眼不等于查一遍。`--no-peek` 可关。
（自带示例库上基本不触发：枚举列都写在 `catalog.yaml` 里，模型有取值可依；
想试就给库加一个有枚举但不在词典里的列，再问"只看 XXX 的……"。）

**为什么口径要能沉淀**：模型不知道你们公司的"大单"指多少钱、退货率怎么算，
而这类知识问一次答错、下次还会答错。`teach` 把「问题 → 正确 SQL」存下来，
下次问到同类问题先注入——实测教一次 `amount >= 10000`，然后问「大单主要来自哪些品类」
「华南的大单有多少笔」「大单的平均金额」，三个不同的问题都会自动带上这个阈值。

**空结果与取值说明为什么值得单独做**：`SELECT SUM(amount)` 在没有匹配行时返回的是
一行 `NULL` 而不是 0 行，所以"有没有数据"不能只看行数；而模型遇到库里没有的取值
（比如问"办公用品"）会自己挑个最近的类别（"配件"）给出金额——这类错误不报错、
不返空，数字看起来完全正常。本项目对前者单独判定、对后者在生成前拦截。

**更隐蔽的一种**：空结果触发的"放宽条件"重试，实测会把「2024 年 3 月」改成「2025-03」，
拿回 100 行真实数据当答案交出去——格式、数字、列名全对，用户根本不会怀疑。
所以重试结果要先过范围守卫：用户原话里的年份、季度、半年、月份、枚举取值必须还在，
否则丢掉结果、如实说明（见 `app/tools/scope_guard.py`）。

**"为什么变了"为什么要单独一条路**：归因问题的答案不是"一个数"，而是"变化来自哪几项"。
让模型直接算占比，很容易出现三项占比加不到 100% 这种没人看得出来的错，
所以这条路径上模型只负责写一条「对比列 × 拆解维度 × 指标」的交叉表，
差分、贡献占比、排序、闭合校验全部由 `app/tools/attribution.py` 用代码算：

```
为什么 6 月的销售额比 5 月高？
→ SQL: SELECT 月份, 品类, SUM(amount) FROM ... WHERE 月份 IN ('2025-05','2025-06') GROUP BY 1, 2
→ 代码拆解：
   「6月」的销售额为 1,854,355.65，比「5月」的 1,408,705.85 增加 445,649.80（+31.6%）。
   主要拉动（3 项，合计 +438,159.25）：
     · 笔记本 +215,648.55（基期 674,130.75 → 本期 889,779.30），占增幅的 48.4%
     · 平板   +201,436.35（基期 268,518.10 → 本期 469,954.45），占增幅的 45.2%
   校验：5 个品类分项的变化加总 = +445,649.80，与总差额一致，可逐项核对。
```

拆不出来就不拆：结果里没有可用的分类列时，明确说"没能拆出归因"，退回普通明细，
而不是硬给一个看起来像结论的百分比。

**想再往下拆一层**：在问题里点两个维度就行，比如"按品类和客户等级拆"。
这条请求会一次取回四列（对比列 × 一级 × 二级 × 指标），代码对**变化最大的那一项**下钻：

```
为什么 6 月的销售额比 5 月高？按品类和客户等级拆
→ 一级（品类）：笔记本 +215,648.55（占增幅 48.4%）、平板 +201,436.35（45.2%）…
→ 二级（笔记本 × 客户等级）：战略客户 +113,559.00（25.5%）、普通客户 +72,821.20（16.3%）…
   校验：以上分项加总 = +215,648.55，与「笔记本」本身的变化一致。
```

二级占比沿用"对总额的贡献"这一口径，所以它加到的是该项自己的占比（48.4%），不是 100% ——
结论里会写一句提醒。最多拆两层：第三层的阅读成本已经超过收益。

**时段可以是一段，不只是一个点**："Q2 比 Q1""下半年比上半年"都认。季度、半年在代码里按
`(年, 起始月, 结束月)` 处理，所以能把库里的月份行归到同一侧，也能推出上一期（Q1 → 去年 Q4、
上半年 → 去年下半年，而不是"上个月"）：

```
为什么 2025 年 Q2 的销售额比 Q1 高？
→ Q1 3,991,583.60 → Q2 4,699,593.60（+708,010.00，+17.7%）
   笔记本 +616,398.65（占增幅 87.1%）、手机 -166,686.05（对冲 23.5%）…
   校验：5 个品类分项的变化加总 = +708,010.00，与总差额一致。
```

季度和半年都能写成 `Q2` / `2025Q2` / `第二季度` / `上半年` / `H1`；
自己写的区间也认：`3-5月`、`3 月到 5 月`、`2025年3月至5月`、`3~5月`。

```
为什么 2025 年 3-5 月的销售额比 1-2 月高？
→ 1-2 月 2,693,130.45 → 3-5 月 4,143,691.10（+1,450,560.65，+53.9%）
   笔记本 +805,134.95（占增幅 55.5%）、平板 18.3%、手机 12.4%…
   注：两期自然天数不同（59 天 / 92 天），总额未按天数归一
```

相对时间也支持（参考日默认运行当天，可用 `AGENT_TODAY` 固定）：

```
上个月的销售额是多少？
→ 若今天=2025-07-15：按 2025-06-01 ~ 2025-06-30 查询（显式区间由系统给定）
→ 若今天=2026-11-25：库只覆盖 2025 年，直接说明没有数据，不换成 2025-06 来答
```

注意 `从 5 月到 6 月` 与 `3-5 月` 的读法不同：前者是**方向**（5 月 → 6 月的变化），
后者是**一个区间**。只问一个季度但库里没有上一季度（比如数据从 2025 年开始，却问"Q1 为什么下降"）时，
会直接说明"只有「2025年Q1」，没有「2024年Q4」，没有可比对象"，退回明细而不是硬凑。

**这条链路的已知边界**（写清楚比假装没有好）：
- `DEFER` 靠模型自觉，提示词只能降低概率；如果它坚持用相近字段硬算，代码层目前拦不住
  ——这类"字段级别的替换"（省份→大区）仍需要 schema 与问题的一致性校验，尚未实现。
- 范围守卫覆盖年份、季度、半年、月份（含 `3-5 月` 这类区间）、枚举取值这几类**明确的字面条件**；
  季度与半年都按"多写法比对"——`2025Q2` / `上半年` 字面、逐月展开、日期区间任一命中即放行，
  只拦被换掉的时段。半年比季度多一层：区间常以"另一半年的首月 1 日"为上界
  （`>= '2025-01-01' AND < '2025-07-01'` 里没有 6 月），所以边界要按方向认，
  否则会把正确答案拦掉。区间锚点放行两种等价写法：逐月展开与"上界写成下一个月 1 日"的日期区间，
  只拦被缩窄的情况。
- 归因只在 SQL 路线启用；可以拆一层，也可以在问题里点两个维度拆两层，
  但**最多两层**，且下钻固定针对变化最大的那一项（要拆别的项就在问题里点名）。
- 期次支持"月 / 季度 / 半年 / 自定义区间 / 年"，**周、日、跨年区间（"11 月到次年 2 月"）不支持**；
  只说一期且前一段跨年时（如"3-5 月为什么下降"，前一段是去年 12 月–今年 2 月）不做归因，
  退回普通查询 —— 当前时段表示法写不出跨年区间，拼一个起止颠倒的假区间比不做更糟。
- **相对时间已由系统锚定**：`上个月` / `最近 30 天` / `本季度` 等按**运行当天**换算成显式日期区间，
  不再让模型猜"今天"。若这段时间与库覆盖没有交集，直接说明没有数据、**不会平移到有数据的时段**。
  要固定口径（历史回放、评测）可设环境变量 `AGENT_TODAY=2025-07-15`。
  只认公历自然周期；"月底""季度末"和跨年区间（"11 月到次年 2 月"）不换算。

## SQL 复用：同一个问题问第二遍，不再调模型

看板刷新、核对同一个数、脚本定时跑同一批问题 —— 真实使用里"同一个问题问很多遍"很常见。
参数是一样的，让模型再猜一次既慢又可能猜出不一样的写法。现在这类重复直接复用上次跑通的 SQL：

```
$ python -m app.cli ask "各区域的销售额是多少？"       # 第一次
[路由] SQLite 多表（Text-to-SQL）
[生成的SQL] SELECT c.region AS 区域, ROUND(SUM(o.amount), 2) AS 销售额 ...
[洞察] 华南以 5,064,245.40 元居首 ...
（2 次模型调用 · 1,761 输入 token · 3.31 秒）

$ python -m app.cli ask "各区域的销售额是多少？"       # 第二次
[复用] 查询与结果都与上次一致，结论一并复用
[分析结果]（5 行）…（同样的数字）
（0 次模型调用 · 0 token · 1.04 秒，链路只剩 cache → execute → answer）
```

**结果不会"复读"成旧的**：复用的是查询，执行的是当前库。只有结果表与上次一字不差时，
上次的结论才一起复用；数据变了就只复用 SQL、结论重新生成。要是这条 SQL 已经跑不通
（表被删、字段改名），条目当场作废、重新生成，**不会顺手换到 pandas 路线** ——
那是"模型理解错"的处置，这里错的是缓存本身。

**哪些情况绝不复用**（每条都对应一种"复用就答错"）：

| 情况 | 原因 |
| --- | --- |
| 多轮追问 | 同一句话在不同上下文里指向的可能是不同的东西 |
| 归因问题 | 要的是"两期 × 维度"交叉表，对比期不在问题原文里 |
| 取值库里没有 | 该走澄清，不能让一条旧 SQL 把澄清盖掉 |
| 时间完全超出覆盖 | 该说明"没有数据"，不能拿旧 SQL 算个别的区间 |

缓存键 = 问题（归一化）+ 库结构指纹 + 词典口径指纹 + **相对时间的换算区间**。
最后一项是关键：`上个月` 的 SQL 里是显式日期，键里绑"今天是哪天"会在同月内换天时白白失效，
绑"换算出来的区间"两头都对（1 月 15 日和 1 月 20 日都指 2025-12，可复用；2 月就不是了）。

缓存写在 `~/.data-agent/sqlcache/<库指纹>.json`，单库上限 200 条、按最久没用淘汰。
`--no-cache` 关掉；`teach` / `lessons --rm` 会清空当前库的缓存（用户既然改了这类问题的口径，
旧 SQL 就不该再被复用）。`-f json` 里的 `cache.hit` 能判断这次是复用还是现写。

## 安全说明（重要）

本项目按「本地单机、可信数据」设计，做了能落地的防护，但**不是强隔离沙箱**。上线到公网前请读完本节。

**SQL 路径（较严格）**

- 只读连接：`file:...?mode=ro`，写操作在连接层就被 SQLite 拒绝；
- 只放行单条 `SELECT` / `WITH`，先剥注释再校验，防止用注释夹带关键字；
- 14 个修改/危险关键字黑名单（`insert`/`drop`/`attach`/`pragma`/`load_extension` 等）；
- 查询超时（默认 10 秒，`AGENT_SQL_TIMEOUT` 可调）：用 SQLite progress handler 中断，挡误写的笛卡尔积；
- 结果行数上限（默认 200 行/次，`sql_tool.MAX_ROWS_DEFAULT`；MCP 侧默认截到 200 行并标注 `truncated`）。

**pandas 路径（尽力防护，非强隔离）**

- **AST 静态校验**：禁止 `import`、禁止访问下划线开头的内部属性（挡 `().__class__.__bases__` 逃逸链）、
  禁止 `to_csv`/`to_excel`/`to_sql` 等写盘方法、禁止 `eval`/`exec`/`open`/`getattr` 等名字；
  代码复杂度上限 4000 个语法节点（`AGENT_MAX_CODE_NODES`）；
- **受限命名空间**：只注入白名单内置函数，`df` 传副本，原始数据不会被改；
- **结果行数硬上限** 20000 行（`AGENT_MAX_RESULT_ROWS`），超出截断并告知；
- **执行超时**（默认 20 秒，`AGENT_EXEC_TIMEOUT` 可调）：代码跑在守护线程里，死循环不会卡死会话。

  > 已知局限：Python 无法强杀线程。若生成代码是不释放 GIL 的 C 层紧密计算（如 `sum(range(10**8))`），
  > 主线程要等它跑完才恢复。死循环、pandas/numpy 长任务这类真正会「卡死会话」的场景不受影响。
  > 需要强隔离请用下面的 Docker 方案，并对容器做 CPU / 内存限制。

**通用**

- 单次提问上限 2000 字符（`AGENT_MAX_QUESTION_CHARS`），运行日志只记问题前 200 字与指标，不落原始数据；
- 网页端默认每个会话最多 50 次提问（`AGENT_MAX_QUESTIONS` 可调），防止公开部署被刷额度；
- `.env` 已被 `.gitignore` 忽略；云端部署用平台 secrets 或环境变量，不要提交密钥。

### 可调参数一览（环境变量）

上面这些限额都写在 `app/config.py`，**都能用环境变量覆盖**，不用改代码。
按机器性能或数据规模调整时改这张表里的名字即可（写进 `.env` 或容器环境变量）：

| 环境变量 | 默认 | 作用 | 什么时候改 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | 模型密钥，空则命令行拒绝启动 | 必填 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | 接口地址（OpenAI 兼容） | 换兼容网关时 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名 | 想换更便宜/更强的模型时 |
| `AGENT_TODAY` | 运行当天 | 相对时间的参考日（`YYYY-MM-DD`） | 历史回放、评测、复现「上个月」 |
| `AGENT_CACHE` | 开启 | 设 `0` 关闭 SQL 复用（等同 `--no-cache`） | 批量跑评测、怀疑缓存时 |
| `AGENT_HOME` | `~/.data-agent` | 会话 / 缓存 / 修正记录的根目录 | 想隔离多套环境时 |
| `AGENT_CATALOG` | `app/schema/catalog.yaml` | 业务术语词典路径 | 接自己的库时 |
| `AGENT_MAX_QUESTION_CHARS` | `2000` | 单次提问字符上限 | 需要贴长文本时 |
| `AGENT_MAX_CODE_NODES` | `4000` | 生成代码的 AST 节点上限 | 复杂分析被误拦时 |
| `AGENT_EXEC_TIMEOUT` | `20` | pandas 代码执行超时（秒） | 大数据量算不完时 |
| `AGENT_SQL_TIMEOUT` | `10` | 单条 SQL 执行超时（秒） | 大表聚合超时时 |
| `AGENT_MAX_RESULT_ROWS` | `20000` | pandas 结果的硬行数上限 | 结果被截断时 |
| `AGENT_MAX_QUESTIONS` | `50` | 单个网页会话提问上限，`0` 不限 | 公开部署时调小 |
| `AGENT_LOG_DIR` | `output/logs` | 运行日志目录 | 容器里要挂载日志卷时 |

> 改完可以用 `python -m app.cli schema` 或随便问一句确认生效；
> `tests/test_deploy.py` 会盯着配置读取的优先级（环境变量 → `.env` → `st.secrets`）。

## 测试

离线测试（**不调用 LLM，无需 API Key**）：

```bash
python -m unittest discover -s tests -v
```

共 592 个用例，覆盖：

| 测试文件 | 内容 |
| --- | --- |
| `tests/test_smoke.py` | 代码安全检查、SQL 注入与绕过拦截、Schema 生成、选图规则、结果压缩、上下文加载 |
| `tests/test_graph.py` | 状态机正常路径、SQL 失败换 pandas、错误回喂、重试耗尽、无数据库降级、空结果自愈、聚合空不重试、`COUNT=0` 仍算有效答案（全部 mock LLM） |
| `tests/test_guard.py` | 取值校验正反例（未知取值必拦、泛指维度与固定搭配绝不误拦）、澄清分支端到端（不调用模型、列出真实可选值） |
| `tests/test_scope.py` | 范围守卫：年份/月份/取值被改掉必拦、同月换日期区间不误伤、区间作为一个锚点（展开与日期区间两种写法都放行、缩窄要拦）、季度锚点（字面/逐月/日期区间/裸数字四种写法放行，缩窄与换季必拦，与归因的季度正则对拍）、半年锚点（逐月展开/日期区间/只卡上界/裸数字/字面都放行；换成全年、换成某个季度、单月、上下半年互换、换到上一年必拦；与归因的半年正则对拍）、`DEFER` 与自然语言拒答不执行、改条件换来的数据被丢弃 |
| `tests/test_time_anchor.py` | 相对时间锚点：上个月/最近 N 天/季度/半年（今年上半年、去年下半年、前年 H2）按系统给定参考日换算；无交集直接拒答且不调 LLM（两引擎各一用例 mock 验证）、长词优先（半年不吃掉「今年」）、裸「上半年」不猜年份、裸数字与 `LIMIT 6` 不误判 |
| `tests/test_lesson.py` | 修正回流：入库校验（非 SELECT / 跑不通 / 超长一律拒）、按库指纹隔离、相关度正反例（口径词命中 vs 区域/取值/年份冲突）、注入位置与开关 |
| `tests/test_attribution.py` | 指标归因：语序决定基期（`A比B` / `从A到B`）、单时段默认上一期、跨年只取最新年、合计行剔除、单边缺失记 0、比率指标不可加总、拆不出维度时降级、普通问题不误判、二级拆解（维度顺序、下钻最大项、闭合校验、两层只给一层时的说明）、季度/半年/自定义区间（写法识别、与月份行按交集归并、跨季度、区间外月份不混入、`从 A 到 B` 仍是方向、跨年前一段不做归因、缺基期时说明缺哪一期） |
| `tests/test_ui.py` | 界面首屏渲染、问答结果渲染（表格/图表/下载）、失败告警、拒答走正文而非告警、多轮改写与历史传递、追问建议按钮、追踪号展示（全部 mock LLM） |
| `tests/test_eval.py` | 评测集自洽性（金标准 SQL 全部可执行）、判分器正反例（含列顺序无关与量纲宽容）、refuse/clarify 正反例（给了数就判错） |
| `tests/test_security.py` | AST 逃逸拦截、执行超时、结果截断、SQL 查询超时、日志落盘 |
| `tests/test_deploy.py` | 密钥读取优先级（env → .env → st.secrets）、配置总闸（没 Key / 占位 Key 都必须拦住）、容器文件与忽略规则、Streamlit 配置解析 |
| `tests/test_cli.py` | 旧命令兼容、四种输出格式、导出 csv/json/md/xlsx、会话存取与坏文件容错、日志读取、`logs --trace` 参数、追问建议、退出码 |
| `tests/test_trace.py` | 追踪号格式与同秒不撞号、阶段事件不混入汇总行、完整号/前缀/尾号三级匹配与多链提示、状态机的重试环与澄清分支、线性路线的顺利/回退/全失败三条链、节点异常留痕、字段白名单（只记长度/行数、零值不写、未知键丢弃）、CLI 链路输出 |
| `tests/test_mcp_server.py` | MCP 工具注册与参数 schema（默认值要落在省 token 那一档）、`ask` 的开关映射（`plan=False` 真的落成 `--no-plan`）、截断标注（`rows` 是真实总数、`data` 只有 N 行、`truncated` 为真）、中文不转义、空问题/库不存在/模型异常的三种结构化错误、`dry_run` 绝不调模型、草稿落到库文件同目录且能被 `yaml.safe_load` 读回、**工具调用期间 stdout 必须为空**（stdio 协议通道）、未装 mcp 时的提示语、`serve --db/--name` 真的传下去 |
| `tests/test_prompt_budget.py` | 提示词预算：schema / 术语在整份 prompt 里只出现一次、示例消息里一次都没有、模板不许再写回 `{schema}` / `{terms}`、新写法必须比老写法短 40% 以上（把占位符改回去会红光） |
| `tests/test_catalog_gen.py` | 词典自动生成：类型映射（含"TEXT 存日期"与"没声明类型"）、枚举四条判据与标识符排除（回归 `customer_id` 误认）、外键声明与推断、脏 JSON 解析、不存在的表列被丢弃、三方合并优先级（已有 > 模型 > 实测，事实除外）、YAML 往返与特殊字符转义、参数超长拒绝、草稿喂给 `build_db_schema_text`、草稿激活 `enum_guard` |
| `tests/test_peek.py` | 两段式探查：`PEEK:` 前缀（含全角冒号、大小写、二次探查）、白名单正反例（单表/纯列名放行，JOIN/UNION/子查询/聚合/`*`/多表/多语句全拒）、单列压成取值清单与多列按行回灌、空值与超长值、额度用完与死循环、探查失败原因回灌、两引擎端到端 |
| `tests/test_plan.py` | 计划先行：何时才出计划（过滤语义/比率指标/长问题触发，简单聚合与并列问法不触发）、从模型输出里抠 JSON（代码块、前后废话、多个花括号）、表名与列名只提醒不阻断、时间与库里数据无交集必拦、取值不在枚举必拦、年份月份与原话不符必拦、归因问题跳过时间校验、致命问题分流到拒答/澄清、失败退回普通生成、两引擎都传得到计划提示 |
| `tests/test_sql_cache.py` | SQL 复用：键的归一化（标点空白无关）与隔离（库/口径/相对时间区间）、相对时间绑"换算后的区间"（同月内换天仍命中、换月必失效）、七条不该复用的规则（多轮/归因/未知取值/时间超范围/单表/关闭/未命中）、命中计数与 LRU 淘汰、同问题覆盖不堆积、坏文件与**手改注入的 `DROP` 被拒**、`teach` 清空缓存、结果指纹（同数据同指纹/变数据变指纹/空表）、状态机命中后**零模型调用**、结论一并复用、数据变了重算结论、过时 SQL 作废后重新生成且**不换 pandas**、线性引擎同样复用 |

### 持续集成

推送到 GitHub 后自动跑同一套测试（`.github/workflows/tests.yml`，**放在仓库根**）：

| 作业 | 做什么 |
| --- | --- |
| `offline-tests` | Python 3.12 与 3.13 双版本；按 `pip install -r requirements.txt` 安装（就是你要用的那条命令）；跑全部离线用例 + `evals.run_eval --dry-run` |
| `docker-image` | 构建镜像 → 断言镜像里没有 `.env` → 在容器里再跑一遍同一套测试 |

**CI 里刻意不注入任何密钥**，并额外做一次静态扫描（不许有 `.env`、不许有 `DEEPSEEK_API_KEY`、
扫一遍 `sk-` 长串）。这样既不会有云端账单，也反过来证明这些用例真的不联网。

> 想在自己机器上模拟 CI（没有任何本机配置）：
> `git archive HEAD data-bi-agent | tar -x -C /tmp/ci-sim` 然后进去跑测试即可。

## 评测

评测集用「问题 + 金标准 SQL」的方式建：标准答案由 SQL 在同一个库上现算，不手写，避免抄错；
判分只比结果值，不比列名与列顺序，因此模型用什么别名都不影响对错。

另有 9 条**没有标准答案**的用例，判的是「该不该给数」：
5 条该拒答（问 2024 年但库只有 2025、毛利率但无成本字段、按省份但只有大区、退货率但无退货表）、
4 条该澄清（办公用品/电脑/东北/临时客户，库里都没有这些取值）。只要给出带值的结果就判失败。

```bash
# 1. 先校验评测集（只跑金标准 SQL，不调用模型、不花钱）
python -m evals.run_eval --dry-run

# 2. 跑全量 53 条（会真实调用模型）
python -m evals.run_eval --tag baseline

# 3. 只跑某几条 / 某个分类
python -m evals.run_eval --only A01,B01,K01,J01
python -m evals.run_eval --today 2026-11-25  # 固定相对时间的参考日
python -m evals.run_eval --category 该拒答

# 4. 带上已沉淀的修正记录跑（测"用了一段时间之后"的表现）
python -m evals.run_eval --with-lessons
```

> 评测**默认不加载**修正记录（`use_lessons=False`）：基线要反映冷启动能力，否则一旦 teach 过几条，
> 历史数字就再也比不了。`--with-lessons` 是另一个口径，两者不要混着看。

最近一次基线（`deepseek-chat`，53 条，洞察与图表关闭，**计划先行默认开启、SQL 复用关闭**）：

| 指标 | 结果 |
| --- | --- |
| 准确率 | **53/53（100%）** |
| 执行成功率 | 43/43（100%，仅统计有标准答案的题） |
| 拒答/澄清正确率 | 10/10（100%，该说不的时候没编数） |
| 一次通过率 | 52/53（98.1%，1 条经重试后正确） |
| 平均耗时 | 1.08 秒/题；P95 2.46 秒 |
| 模型调用 | 64 次，输入 **90,557** / 输出 4,540 tokens |

> 与 0.9.2 相比调用 46 → 63 次、token +22%：43 条有答案题里有 10 条触发了查询计划
> （比率/派生指标、多层限定、多轮追问），每条多一次约 2.5k 输入的调用。
> 0.11.0 再 +6.4%（168,292 → 179,084），全部来自提示词里新增的探查规则 ——
> **53 条用例里 0 条触发探查**，因为示例库的枚举列都写在 `catalog.yaml` 里，模型有取值可依。
> 准确率始终是 53/53。在意成本就用 `--no-plan` / `--no-peek`。
>
> 0.12.0（词典自动生成）与 0.13.0（MCP server）**对这条链路零影响** —— 前者是离线命令，
> 后者只是把 `-f json` 的输出抽成共享模块并加了 `serve`，都没改提示词、没加调用：
> 53/53、63 次调用、token 三次持平（179,084 → 179,066 → 179,065 是同一次生成的正常波动）。
> 它的收益体现在"接入新库"那一步，对比见上面 `catalog` 小节。
>
> 0.13.1（半年纳入范围守卫）同样零影响：只动了守卫代码，提示词一字未改，53/53、63 次调用、
> 179,063 tokens。评测集里两条半年题本来答对、也没触发过重试，守卫在那批题上是待命状态，
> 它的正反例行为由 `tests/test_scope.py` 的 12 条用例盯着。
>
> **0.14.0 把这条链路的输入成本砍了一半**：提示词里 schema 与术语原来被每条 few-shot 各带一份
> （1299 + 198 字符重复 4 次 = 整份 prompt 的 82%），现在只在 system 消息里注入一次 ——
> 输入 **179,063 → 90,536 tokens（-49.4%）**，平均耗时 1.24 → 1.10 秒，准确率仍 53/53
> （拒答 6/6、澄清 4/4 不变）。放 system 而不是最后一条消息，是因为 system + 示例是跨问题
> 不变的前缀，DeepSeek 的上下文缓存只在这类前缀上命中。
> 回归由 `tests/test_prompt_budget.py` 盯着：把 `{schema}` 写回模板会直接红灯。
>
> **0.15.0（SQL 复用）不改变这条链路的任何一个数字**：53/53、64 次调用、90,557 tokens
> （与 90,536 差 0.02%，模型波动）。评测里刻意把复用关掉 —— 同一批问题会跑第二遍，
> 命中缓存等于让上一轮的答案给这一轮打分，准确率虚高、token 也没法和历史基线比。
> 复用的收益在"同一个问题第二次提问"上：2 次调用 1,761 token 3.31 秒 → **0 次调用 0 token 1.04 秒**。

用途例覆盖十一类：总量指标（5）、维度汇总（7）、排名 Top-N（5）、条件筛选（6）、业务口径（4）、
时间趋势（3）、复杂推理（8，含环比增长率、分组 Top-1、周末对比）、多轮追问（3）、口语表达（2）、
该拒答（5）、该澄清（4）。

> **怎么读这个 100%**：它说明这套数据上的确定性查询类问题已经稳定，**不代表开放域提问都能答对**。
> 这批题目偏基础、且 schema 只有 3 张表；表变多、口径变模糊时准确率会下降。
> 评测集的价值在于「改 prompt / 换模型后能立刻看出有没有退化」，而不是拿满分当结论。
>
> 9 条拒答/澄清用例是补上"编答案"这个盲区后才加的：加之前同样这 5 个"答不了"的问题里，
> 4 个会被编出一个看着正常的数字（把 2024 改成 2025、拿标价当成本、把省份当大区）。
> 单看准确率没有变化，但这 9 条让"编数"这类错误从此会被判错。
> 另：本次全部一次成功，重试链路未被触发，其行为由 `tests/test_graph.py`、`tests/test_scope.py` 覆盖。

## 部署

### 方式一：Streamlit Community Cloud（最快，免费）

1. Fork / 推送本仓库到 GitHub；
2. 打开 [share.streamlit.io](https://share.streamlit.io)，新建 App，指向仓库与 `app/main.py`；
3. 在 **App settings → Secrets** 里填：

   ```toml
   DEEPSEEK_API_KEY = "sk-你的密钥"
   DEEPSEEK_MODEL = "deepseek-chat"
   ```

   代码会按「环境变量 → .env → st.secrets」的顺序读配置，云端无需 `.env` 文件。
4. 部署完成后即可通过链接访问。模拟数据随仓库分发，无需额外准备。

> 注意：Streamlit Cloud 的免费实例是公开的，任何人都能用你的 Key 提问。
> 建议把 `AGENT_MAX_QUESTIONS` 调小（如 20），并在 DeepSeek 后台设置额度上限。

### 方式二：Docker（推荐用于服务器）

```bash
docker build -t data-bi-agent .
docker run --rm -p 8501:8501 \
  -e DEEPSEEK_API_KEY=sk-你的密钥 \
  --cpus=1 --memory=1g \
  data-bi-agent
```

打开 http://localhost:8501 。镜像内置中文字体，健康检查走 `/_stcore/health`。

`--cpus` / `--memory` 不是可选项：pandas 执行路径不是强隔离沙箱，
限制容器资源能兜住「生成代码把机器跑满」的最坏情况。

### 方式三：本机直接跑

```bash
streamlit run app/main.py
```

默认 http://localhost:8501 ，密钥从 `.env` 读取。

### 部署前检查清单

- [ ] 密钥没有写进代码或提交历史（`git grep "sk-"` 应只命中 `.env.example` 的占位符）
- [ ] `AGENT_MAX_QUESTIONS` 已按公开程度设置
- [ ] 平台侧已设额度/告警，避免被刷
- [ ] 数据是模拟数据；换真实数据前确认脱敏与访问权限

## 设计参考

借鉴了以下开源项目的工程实践（**未复制任何代码**，仅借鉴架构与交互设计）：

- **校验前置**（OpenChatBI）：生成的代码先做静态校验，通过后才执行
- **只发 Schema 不发全表**（PandasAI）：更省 token，原始数据不出本地
- **指标口径语义层**（WrenAI / OpenChatBI）：把「客单价 = 销售额 / 订单数」这类口径写进 `app/schema/catalog.yaml`，避免模型自行发挥
- **计划先行**（WrenAI 的 `plan → validate → execute`、fnauman/txt2sql 的语义规划）：写 SQL 前先摊开"我理解成了什么"，
  validate 那一步由代码做而不是再问一次模型
- **两段式探查**（Vanna 的 `intermediate_sql`）：允许模型先要一次受限只读查询看一眼真实取值，
  再写最终 SQL。只借机制，没照搬"任意只读查询"——白名单与额度都是本项目收窄的
- **词典描述自动生成**（XiYan-SQL 的 `DBDescGen`）：只借"接新库自动出描述"这个目标，
  分工反着来 —— 代码实测事实、模型只补语义、人拍板（见 `catalog` 小节）
- **对外协议**（Model Context Protocol）：把 `ask` / `get_schema` / `generate_catalog` 包成
  stdio server，别的 Agent 可直接调用。借的是协议本身，工具边界（只读、截断、结构化错误）
  是按"返回值会进另一个模型的上下文"这条约束重新划的
- **命令行形态**（natural-sql / nlsql / simonw-llm）：子命令结构、`--format` 多格式输出、
  `chat -c` 续聊、`logs` 查询运行记录、stdin 管道、语义化退出码

## 版本管理

`python -m app.cli version` 会打印当前版本；每个版本改了什么、有什么行为差异，
见 [CHANGELOG.md](CHANGELOG.md)。

按功能粒度提交：每完成一个可验证的功能点即提交一次，保证任何一个提交点都能跑起来。

提交信息使用约定式前缀：

| 前缀 | 用途 |
| --- | --- |
| `feat:` | 新功能 |
| `fix:` | 修复缺陷 |
| `refactor:` | 重构（不改变行为） |
| `test:` | 测试 |
| `docs:` | 文档 |
| `chore:` | 依赖、配置等杂项 |

> 提交前请确认 `git status` 中没有 `.env`。

## 许可

本项目采用 [MIT License](LICENSE)。

## 贡献

欢迎 Issue / PR。提交前请确保：
1. `python -m unittest discover -s tests -v` 全部通过；
2. `python -m app.cli "各区域的销售额是多少？"` 可正常返回。
