"""运行日志（S5.3）。

为什么单独一个模块：
- 日志写盘是横切关注点，塞进各节点会污染业务逻辑；
- 统一成一行一条 JSON（JSONL），方便用 `jq`、Excel、pandas 直接分析；
- 只记关键字段（问题、路由、重试、耗时、结果规模、错误），不记 SQL 全文与用户隐私数据。

日志位置：`output/logs/agent.jsonl`（`output/` 已被 .gitignore 忽略，不会进仓库）。
单文件超过 2MB 自动轮转，最多保留 3 份历史，长期运行不会占满磁盘。
"""
import json
import logging
import logging.handlers
import os
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

from app.config import LOG_DIR

LOGGER_NAME = "data_bi_agent"
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 3

_logger = None


def get_logger() -> logging.Logger:
    """懒初始化日志器（带上按大小轮转的文件处理器）。"""
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                os.path.join(LOG_DIR, "agent.jsonl"),
                maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8",
            )
        except Exception:
            # 目录不可写（只读部署、权限受限）时降级为不落盘，不能影响主流程
            handler = logging.NullHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    _logger = logger
    return logger


def log_run(**fields) -> None:
    """记录一次问答。任何异常都不能影响主流程，所以整体兜住。"""
    try:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
        get_logger().info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        pass


class TokenCounter(BaseCallbackHandler):
    """统计一次问答的模型调用次数与 token 用量（成本监控用）。

    必须继承 BaseCallbackHandler：LangChain 会读 ignore_chain / raise_error 等属性。
    通过 LangGraph 的 config 挂到 LLM 调用上，不侵入生成代码。
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self.calls += 1
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if not usage:
            try:
                message = response.generations[0][0].message
                usage = getattr(message, "usage_metadata", None) or {}
            except Exception:
                usage = {}
        self.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


def reset_logger() -> None:
    """关掉并清空日志器（测试用，避免文件句柄占用导致临时目录删不掉）。"""
    global _logger
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
    _logger = None


def log_path() -> str:
    """当前日志文件路径（命令行 `logs` 子命令要显示它）。"""
    return os.path.join(LOG_DIR, "agent.jsonl")


def read_runs(limit: int = 10, failed_only: bool = False,
              path: str = None) -> list:
    """读取最近的运行记录，供 `python -m app.cli logs` 展示。

    - 坏行（截断、手工编辑）直接跳过，不能因为一行脏数据就看不了日志；
    - 倒着读再反转，避免把整个文件读进内存（日志会轮转，但单文件仍可能几 MB）。
    """
    path = path or log_path()
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if failed_only and record.get("ok"):
                continue
            rows.append(record)

    return rows[-limit:] if limit else rows
