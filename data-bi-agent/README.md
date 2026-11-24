# 数据分析 Agent（Text-to-Pandas / Text-to-SQL）

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
```

显示时间、成败、路由、重试次数、结果行数、耗时、token 用量。日志在 `output/logs/agent.jsonl`。

### 公共参数

| 参数 | 作用 | 适用 |
| --- | --- | --- |
| `--db PATH` | 换一个 SQLite 库文件（默认 `data/sales.db`）；路径不存在直接报错退出 | ask / chat / schema |
| `--no-chart` | 不生成图表 | ask / chat |
| `--no-insight` | 不生成洞察文案（省一次模型调用） | ask / chat |
| `--linear` | 用线性编排（无重试、无多轮记忆），便于对比 | ask / chat |
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
| 左侧栏 | 数据源概览（表名 + 行数）、可展开查看随问题发给模型的 Schema、图表/洞察开关、6 个示例问题、清空对话 |
| 对话区 | 一问一答，保留多轮上下文，支持「只看华东」这类追问 |
| 结果区 | 中文洞察结论、结果表格（可下载 CSV）、交互式图表（折线/柱状/饼图/散点）、可展开查看生成的 SQL 或 Python、**追问建议按钮**（点一下接着问） |
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

## 项目结构

```
app/
├── config.py            # 配置与密钥读取（环境变量 / .env / st.secrets）+ 安全限额
├── llm.py               # LLM 客户端与代码块清洗
├── context.py           # 运行时上下文（数据/Schema/术语，惰性加载）
├── state.py             # AgentState 状态结构
├── graph.py             # LangGraph 状态机（含重试环）+ 运行日志
├── logging_utils.py     # 运行日志与 token 计数（成本监控）
├── agent.py             # 线性编排（--linear 时使用）
├── cli.py               # 命令行入口（子命令 ask / chat / schema / logs / version）
├── render.py            # 结果渲染与导出（table / markdown / csv / json / xlsx）
├── sessions.py          # 会话持久化（~/.data-agent/sessions，支持 chat -c 续聊）
├── main.py              # Streamlit 网页界面（聊天式问答）
├── schema/
│   ├── loader.py        # 数据加载 / Schema 描述（DataFrame 版 + 数据库版）
│   └── catalog.yaml     # 业务语义词典（字段含义、指标口径、枚举值）
├── prompts/templates.py # Prompt 模板（代码/SQL/洞察/问题改写/错误修正）
├── nodes/
│   ├── understand.py    # 问题理解：多轮改写 + 路由判定
│   ├── generate.py      # Text-to-Pandas / Text-to-SQL 生成
│   ├── execute.py       # Python 代码安全执行（AST 校验 + 超时 + 截断）
│   ├── reflect.py       # 反思重试与换路
│   └── answer.py        # 结果压缩 + 洞察生成
└── tools/
    ├── sql_tool.py      # SQL 安全校验与只读执行（含查询超时）
    ├── chart_tool.py    # 规则式自动选图（Plotly）
    ├── suggest_tool.py  # 追问建议（规则式，零 token）
    └── pandas_tool.py   # 预留
data/
├── sales.db             # 模拟销售库（SQLite，3 张关联表）
├── sales_orders.csv     # 模拟销售宽表
└── generate_mock_data.py # 模拟数据生成脚本
evals/
├── cases.yaml           # 43 条评测用例（问题 + 金标准 SQL + 判定方式）
├── judge.py             # 判分器（数值容差、列名/列顺序无关）
├── run_eval.py          # 跑评测并产出报告
└── reports/             # 评测报告（仅 baseline 入库，临时报告已忽略）
tests/
├── test_smoke.py        # 离线冒烟测试（34 个用例）
├── test_graph.py        # 状态机与重试测试（9 个用例，全 mock LLM）
├── test_ui.py           # 网页界面测试（4 个用例，全 mock LLM）
├── test_eval.py         # 评测集与判分测试（22 个用例）
├── test_security.py     # 安全防护测试（16 个用例）
├── test_deploy.py       # 部署配置测试（9 个用例）
└── test_cli.py          # 命令行与新模块测试（50 个用例，全离线）
Dockerfile               # 单镜像部署（含中文字体与健康检查）
.streamlit/config.toml   # 界面主题与服务器配置
```

## 工作流程

```
问题 + 历史对话
 ↓
understand：有历史则改写成完整问题；判定走 SQL 还是 pandas
 ↓
generate：生成 SQL 或 pandas 代码（重试时带上一次的真实报错）
 ↓
execute：SQL 走只读库 / 代码走受限命名空间
 ↓
  ├─ 成功且有数据 ─→ answer：规则选图（Plotly HTML）+ 统计分析 → 中文洞察
  ├─ 成功但 0 行 ─→ reflect：不换路，只提示"放宽条件 / 核对枚举值"重写一次
  │                   → 仍为空则如实说明"没有匹配数据"，不空转、不编结论
  └─ 失败 ────────→ reflect：记录错误、SQL 失败则换 pandas → 回到 generate（最多 2 次）
```

## 安全说明（重要）

本项目按「本地单机、可信数据」设计，做了能落地的防护，但**不是强隔离沙箱**。上线到公网前请读完本节。

**SQL 路径（较严格）**

- 只读连接：`file:...?mode=ro`，写操作在连接层就被 SQLite 拒绝；
- 只放行单条 `SELECT` / `WITH`，先剥注释再校验，防止用注释夹带关键字；
- 14 个修改/危险关键字黑名单（`insert`/`drop`/`attach`/`pragma`/`load_extension` 等）；
- 查询超时（默认 10 秒）：用 SQLite progress handler 中断，挡误写的笛卡尔积；
- 结果行数上限（默认 200 行/次）。

**pandas 路径（尽力防护，非强隔离）**

- **AST 静态校验**：禁止 `import`、禁止访问下划线开头的内部属性（挡 `().__class__.__bases__` 逃逸链）、
  禁止 `to_csv`/`to_excel`/`to_sql` 等写盘方法、禁止 `eval`/`exec`/`open`/`getattr` 等名字；
  代码复杂度上限 4000 个语法节点；
- **受限命名空间**：只注入白名单内置函数，`df` 传副本，原始数据不会被改；
- **执行超时**（默认 20 秒）：代码跑在守护线程里，死循环不会卡死会话。

  > 已知局限：Python 无法强杀线程。若生成代码是不释放 GIL 的 C 层紧密计算（如 `sum(range(10**8))`），
  > 主线程要等它跑完才恢复。死循环、pandas/numpy 长任务这类真正会「卡死会话」的场景不受影响。
  > 需要强隔离请用下面的 Docker 方案，并对容器做 CPU / 内存限制。

**通用**

- 单次提问上限 2000 字符，运行日志只记问题前 200 字与指标，不落原始数据；
- 网页端默认每个会话最多 50 次提问（`AGENT_MAX_QUESTIONS` 可调），防止公开部署被刷额度；
- `.env` 已被 `.gitignore` 忽略；云端部署用平台 secrets 或环境变量，不要提交密钥。

## 测试

离线测试（**不调用 LLM，无需 API Key**）：

```bash
python -m unittest discover -s tests -v
```

共 145 个用例，覆盖：

| 测试文件 | 内容 |
| --- | --- |
| `tests/test_smoke.py` | 代码安全检查、SQL 注入与绕过拦截、Schema 生成、选图规则、结果压缩、上下文加载 |
| `tests/test_graph.py` | 状态机正常路径、SQL 失败换 pandas、错误回喂、重试耗尽、无数据库降级、空结果自愈与如实说明（全部 mock LLM） |
| `tests/test_ui.py` | 界面首屏渲染、问答结果渲染（表格/图表/下载）、失败告警、多轮改写与历史传递、追问建议按钮（全部 mock LLM） |
| `tests/test_eval.py` | 评测集自洽性（金标准 SQL 全部可执行）、判分器正反例（含列顺序无关与量纲宽容） |
| `tests/test_security.py` | AST 逃逸拦截、执行超时、结果截断、SQL 查询超时、日志落盘 |
| `tests/test_deploy.py` | 密钥读取优先级（env → .env → st.secrets）、容器文件与忽略规则、Streamlit 配置解析 |
| `tests/test_cli.py` | 旧命令兼容、四种输出格式、导出 csv/json/md/xlsx、会话存取与坏文件容错、日志读取、追问建议、退出码 |

## 评测

评测集用「问题 + 金标准 SQL」的方式建：标准答案由 SQL 在同一个库上现算，不手写，避免抄错；
判分只比结果值，不比列名与列顺序，因此模型用什么别名都不影响对错。

```bash
# 1. 先校验评测集（只跑金标准 SQL，不调用模型、不花钱）
python -m evals.run_eval --dry-run

# 2. 跑全量 43 条（会真实调用模型）
python -m evals.run_eval --tag baseline

# 3. 只跑某几条 / 某个分类
python -m evals.run_eval --only A01,B01,G01
python -m evals.run_eval --category 多轮追问
```

最近一次基线（`deepseek-chat`，43 条，洞察与图表关闭）：

| 指标 | 结果 |
| --- | --- |
| 准确率 | **43/43（100%）** |
| 执行成功率 | 43/43（100%） |
| 一次通过率 | 43/43（100%，无重试） |
| 平均耗时 | 0.9 秒/题；P95 1.4 秒 |
| 模型调用 | 46 次，输入 123,337 / 输出 2,359 tokens |

用例覆盖九类：总量指标（5）、维度汇总（7）、排名 Top-N（5）、条件筛选（6）、业务口径（4）、
时间趋势（3）、复杂推理（8，含环比增长率、分组 Top-1、周末对比）、多轮追问（3）、口语表达（2）。

> **怎么读这个 100%**：它说明这套数据上的确定性查询类问题已经稳定，**不代表开放域提问都能答对**。
> 这批题目偏基础、且 schema 只有 3 张表；表变多、口径变模糊时准确率会下降。
> 评测集的价值在于「改 prompt / 换模型后能立刻看出有没有退化」，而不是拿满分当结论。
> 另：本次全部一次成功，重试链路未被触发，其行为由 `tests/test_graph.py` 的 mock 用例覆盖。

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
- **命令行形态**（natural-sql / nlsql / simonw-llm）：子命令结构、`--format` 多格式输出、
  `chat -c` 续聊、`logs` 查询运行记录、stdin 管道、语义化退出码

## 版本管理

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
