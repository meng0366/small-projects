# 数据分析 Agent（Text-to-Pandas / Text-to-SQL）

用自然语言向数据提问，Agent 自动生成并执行分析代码，返回结果表格与洞察。

> 定位：**轻量级、可读、可改**的数据分析 Agent 教学/原型项目。不追求企业级功能，重在把 Text2SQL 的完整链路跑通并保持代码简洁。

## 功能现状

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 环境与配置 | ✅ |
| 1 | 单表问答（Text-to-Pandas，命令行） | ✅ |
| 2 | 多表 + SQL + 图表 + 洞察 | ⬜ 计划中 |
| 3 | LangGraph 编排 + 错误自愈 | ⬜ 计划中 |
| 4 | Streamlit 界面 | ⬜ 计划中 |
| 5 | 评测 / 安全加固 / 部署 | ⬜ 计划中 |

## 环境要求

- Python **3.11+**（已在 3.13 验证）
- 一个 OpenAI 兼容的 LLM API Key（默认 DeepSeek，可换任意兼容平台）
- 无需数据库服务，默认使用本地 SQLite / CSV

## 快速开始

### 1. 获取代码并进入目录

```bash
git clone <your-repo-url> data-bi-agent
cd data-bi-agent
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
# 交互模式
python -m app.cli

# 单次提问
python -m app.cli "各区域的销售额是多少？"
```

> 请在**项目根目录**下执行以上命令。

## 示例问题

- 各区域的销售额是多少？
- 销售额最高的前 5 个商品是什么？
- 每个月的销售额走势如何？
- 哪个品类的客单价最高？

## 项目结构

```
app/
├── config.py            # 配置与密钥读取
├── state.py             # Agent 状态结构
├── cli.py               # 命令行入口
├── graph.py             # LangGraph 状态机（阶段 3）
├── main.py              # Streamlit 界面（阶段 4）
├── schema/
│   ├── loader.py        # 数据加载 / Schema 描述 / 业务术语
│   └── catalog.yaml     # 业务语义词典（字段含义、指标口径）
├── prompts/templates.py # Prompt 模板（系统指令 + few-shot）
├── nodes/
│   ├── generate.py      # Text-to-Pandas 代码生成
│   ├── execute.py       # 代码安全执行
│   ├── understand.py    # 问题理解（阶段 3）
│   ├── reflect.py       # 错误反思（阶段 3）
│   └── answer.py        # 洞察生成（阶段 2）
└── tools/               # SQL / pandas / 图表工具（阶段 2）
data/
├── sales.db             # 模拟销售库（SQLite，3 张关联表）
├── sales_orders.csv     # 模拟销售宽表
└── generate_mock_data.py # 模拟数据生成脚本
tests/
└── test_smoke.py          # 离线冒烟测试
```

## 工作流程

```
问题 → 读取 Schema + 业务术语 → LLM 生成 pandas 代码
     → 安全检查（黑名单 + 语法编译 + 受限命名空间）
     → 执行 → 格式化输出
```

## 安全说明（重要）

- 生成代码通过 `exec` 执行，已做**轻量防护**（危险调用黑名单、受限内置函数、df 副本），但**不是强隔离**。
- 当前版本适合**本地单机、可信数据**场景。请勿直接暴露为公网服务。
- 生产使用前请启用子进程/容器隔离（见路线图阶段 5）。
- `.env` 已被 `.gitignore` 忽略，请勿提交真实密钥。

## 测试

离线冒烟测试（不调用 LLM，无需 API Key）：

```bash
python -m unittest discover -s tests -v
```

## 设计参考

借鉴了 OpenChatBI、PandasAI、WrenAI 三个开源项目的工程实践（**未复制任何代码**，仅借鉴架构思路）：

- **校验前置**（OpenChatBI）：生成的代码先做静态校验，通过后才执行
- **只发 Schema 不发全表**（PandasAI）：更省 token，原始数据不出本地
- **指标口径语义层**（WrenAI / OpenChatBI）：把「客单价 = 销售额 / 订单数」这类口径写进 `app/schema/catalog.yaml`，避免模型自行发挥

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
