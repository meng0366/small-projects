"""全局配置：模型、密钥与安全限额。

配置来源优先级：环境变量 > .env > Streamlit 云端 secrets。
- 本地开发：`.env`（已被 .gitignore 忽略，不会进仓库）；
- 服务器 / Docker：环境变量；
- Streamlit Cloud：App settings → Secrets 里填 `DEEPSEEK_API_KEY = "sk-..."`。

安全限额（阶段 5 加固）都可用环境变量覆盖，便于按机器性能调整。
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _read_setting(name: str, default: str = "") -> str:
    """按 环境变量 → .env → Streamlit secrets 的顺序读取一个配置项。"""
    value = os.getenv(name, "")
    if value:
        return value
    # 只有环境变量与 .env 都没有时才碰 streamlit，避免命令行启动白加载它
    try:
        import streamlit as st

        secret = st.secrets.get(name, "")   # type: ignore[attr-defined]
        if secret:
            return str(secret)
    except Exception:
        pass
    return default


# 默认使用 DeepSeek（OpenAI 兼容接口）
LLM_API_KEY = _read_setting("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = _read_setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = _read_setting("DEEPSEEK_MODEL", "deepseek-chat")

# 数据目录
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 相对时间的参考日（P0 时间锚点）。默认取运行当天；设为 'YYYY-MM-DD' 可固定口径，
# 例如做历史数据回放或让"上个月"始终指某个已知月份，便于复现与测试。
AGENT_TODAY = _read_setting("AGENT_TODAY", "")

# ---------------------------------------------------------------- 安全限额
MAX_QUESTION_CHARS = int(_read_setting("AGENT_MAX_QUESTION_CHARS", "2000"))
"""单次提问的最大字符数，防止超长文本把上下文撑爆。"""

MAX_CODE_NODES = int(_read_setting("AGENT_MAX_CODE_NODES", "4000"))
"""生成代码的 AST 节点上限，挡住异常超长代码。"""

EXEC_TIMEOUT_SECONDS = float(_read_setting("AGENT_EXEC_TIMEOUT", "20"))
"""pandas 代码执行超时（秒），防止死循环把会话卡死。"""

SQL_TIMEOUT_SECONDS = float(_read_setting("AGENT_SQL_TIMEOUT", "10"))
"""单条 SQL 的执行超时（秒），靠 SQLite 的 progress handler 中断。"""

MAX_RESULT_ROWS = int(_read_setting("AGENT_MAX_RESULT_ROWS", "20000"))
"""结果集行数上限，超出截断，避免大结果撑爆内存与前端。"""

LOG_DIR = _read_setting("AGENT_LOG_DIR", os.path.join(OUTPUT_DIR, "logs"))
"""运行日志目录。"""

MAX_QUESTIONS_PER_SESSION = int(_read_setting("AGENT_MAX_QUESTIONS", "50"))
"""单次网页会话的提问上限，0 表示不限制。公开部署时防止被刷爆额度（刷新页面可重置，属软限制）。"""


def check_config() -> bool:
    """检查关键配置是否就绪。返回 True 表示可用。"""
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-在这里"):
        return False
    return True
