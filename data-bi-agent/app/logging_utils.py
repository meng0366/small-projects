"""运行日志（S5.3 / P2-2）。

为什么单独一个模块：
- 日志写盘是横切关注点，塞进各节点会污染业务逻辑；
- 统一成一行一条 JSON（JSONL），方便用 `jq`、Excel、pandas 直接分析；
- 只记关键字段（问题、路由、重试、耗时、结果规模、错误），不记 SQL 全文与用户隐私数据。

日志分两类，靠 `trace_id` 串起来（P2-2）：
- **汇总行**：一次问答一行，`log_run()` 写，回答完才落盘，字段是"结果怎么样"；
- **阶段事件**：每个节点一行，`log_event()` 写，字段 `stage` 标明阶段（understand /
  generate / execute / reflect / answer），用来回答"慢在哪、卡在哪、重试了几次"。
两者共用同一个 `trace_id`，`jq 'select(.trace_id=="...")'` 就能还原完整链路。
汇总行没有 `stage` 字段 —— 这是区分两类行的唯一约定，`read_runs()` 据此过滤。

日志位置：`output/logs/agent.jsonl`（`output/` 已被 .gitignore 忽略，不会进仓库）。
单文件超过 2MB 自动轮转，最多保留 3 份历史，长期运行不会占满磁盘。
"""
import itertools
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
_seq = itertools.count(1)      # 进程内自增：保证同一秒里的多次问答不撞号


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


def new_trace_id() -> str:
    """生成一次问答的追踪号，形如 `20261124-225500-a10001`。

    构造：`时间-进程号低字节 进程内自增序号`，两个设计点都是踩过坑才定的：
    - **不用随机后缀**：第一版用 `secrets.token_hex(2)`，同一秒里连问 200 次实测撞了 1 次
      （4 位十六进制只有 6.5 万种，按生日问题算两百次就有三成概率撞）。
      两条链撞成一个号，比没有 trace_id 更难查 —— 看起来是一条完整链路，其实混了两次问答。
    - **带进程号**：命令行与网页可能同时提问、两个进程的序号都从 1 开始，
      只靠序号会在"同一秒 + 同一个序号"时撞号。
    """
    return (f"{datetime.now():%Y%m%d-%H%M%S}-"
            f"{os.getpid() % 256:02x}{next(_seq) % 0x10000:04x}")


def log_run(**fields) -> None:
    """记录一次问答的汇总行。任何异常都不能影响主流程，所以整体兜住。"""
    try:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
        get_logger().info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        pass


def log_event(trace_id: str, stage: str, **fields) -> None:
    """记录一个阶段事件（P2-2）。

    与 `log_run` 的区别：这是"过程"，一次问答会有多行；汇总行是"结果"，只有一行。
    带上 `stage` 字段即可与汇总行区分开，不需要另立文件（否则排查时要在两个文件间对号）。
    """
    try:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "trace_id": trace_id,
            "stage": stage,
            **fields,
        }
        get_logger().info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        pass


# ---------------------------------------------------------------- 字段白名单（P2-2）
#
# 为什么要有白名单：日志会被翻、会被分享、会被贴进 issue。阶段事件里如果顺手把
# state 整个 dump 进去，SQL 全文、结果表、用户原话就一起进了磁盘。
# 所以这里只放"能回答慢在哪、卡在哪"的标量，且由代码统一过滤 ——
# 状态机（graph）与线性编排（agent）共用这一份，避免两套引擎记出两种口径。

TRACE_FIELDS = (
    # (状态里的键, 日志字段名, 取值转换)
    ("route", "route", str),
    ("retry_count", "retry", int),
    ("statement", "statement_len", lambda v: len(str(v or ""))),
    ("result", "rows", lambda v: 0 if v is None else len(v)),
    ("error", "error", lambda v: str(v or "")[:200]),
    ("unknown_enum", "clarify", bool),
    ("defer", "defer", bool),
    ("scope_violation", "scope_violation", bool),
    ("attribution", "attribution", lambda v: bool((v or {}).get("ok"))),
    ("chart_type", "chart", str),
    ("lessons_used", "lessons", len),
    ("errors", "errors", len),
    ("failed", "failed", bool),
    ("fallback", "fallback", bool),
)


def trace_fields(values: dict = None, **extra) -> dict:
    """按白名单挑出可落盘的字段；`extra` 用于白名单外的少量补充（如 fallback）。

    - **只收标量**：语句原文只记长度，结果只记行数；不认识、没发生的键直接丢；
    - **零值不写**：0 / False / 空串只是噪音，唯一例外是 `rows=0` —— "查不到数据"是结论；
    - 转换失败（类型不对、对象没有长度）就跳过，日志不能因为一个字段写不进去而整条丢失。
    """
    merged = dict(values or {})
    merged.update(extra or {})
    fields = {}
    for key, name, cast in TRACE_FIELDS:
        if key not in merged:
            continue
        try:
            value = cast(merged.get(key))
        except Exception:
            continue
        if value in (0, False, "", None) and name != "rows":
            continue          # rows=0 是有意义的（查不到数据），其余零值只是噪音
        fields[name] = value
    return fields


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


def _iter_records(path: str):
    """逐行读日志，坏行（截断、手工编辑）直接跳过。

    不能因为一行脏数据就看不了日志 —— 日志最常见的用途恰恰是"出问题之后"。
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def read_runs(limit: int = 10, failed_only: bool = False,
              path: str = None) -> list:
    """读取最近的**汇总行**（一次问答一条），供 `python -m cli logs` 展示。

    - 带 `stage` 的是阶段事件，不属于"一次问答"，跳过；否则表格里会冒出
      一堆 generate/execute 行，看着像每次问答都失败；
    - 倒着取尾部而不是读完再切片，避免把整个文件读进内存
      （日志会轮转，但单文件仍可能几 MB）。
    """
    path = path or log_path()
    rows = []
    for record in _iter_records(path):
        if "stage" in record:
            continue
        if failed_only and record.get("ok"):
            continue
        rows.append(record)
    return rows[-limit:] if limit else rows


def read_events(trace_id: str = "", path: str = None) -> list:
    """按追踪号取阶段事件，按写盘顺序返回（也就是执行顺序）。

    匹配顺序：完整号 → 开头前缀 → 任意子串。
    这么设计是因为人抄号码的习惯不一样：有时从表格里复制整段，
    有时只记得尾巴上那几位（`240001`）。查不到比查到多条更糟 ——
    查到多条至少还能看出是哪几条，查不到就以为日志丢了。
    `_show_trace()` 会在命中多条时提醒，不会把两条链悄悄粘成一条。
    """
    path = path or log_path()
    events = [r for r in _iter_records(path) if "stage" in r]
    if not trace_id:
        return events

    for matches in (
        lambda t: t == trace_id,
        lambda t: t.startswith(trace_id),
        lambda t: trace_id in t,
    ):
        found = [r for r in events if matches(str(r.get("trace_id", "")))]
        if found:
            return found
    return []
