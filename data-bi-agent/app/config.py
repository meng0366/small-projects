"""全局配置：从 .env 读取模型与密钥信息。

对应功能 F0.2 配置与密钥管理。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 默认使用 DeepSeek（OpenAI 兼容接口）
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def check_config() -> bool:
    """检查关键配置是否就绪。返回 True 表示可用。"""
    if not LLM_API_KEY or LLM_API_KEY.startswith("sk-在这里"):
        return False
    return True
