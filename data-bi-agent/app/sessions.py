"""会话持久化（阶段 6：命令行续聊）。

借鉴点：`simonw/llm` 的 `llm chat -c`（不带 id 就续上最近一次会话），以及
`nlsql` 把历史提问落盘、下次作为上下文喂回模型的做法。

设计取舍：
- **会话存在用户主目录**（`~/.data-agent/sessions/*.json`），不放仓库 `output/`：
  会话含用户真实提问，属于个人数据，不该跟着项目目录搬家，也不该被误提交；
  根目录可用环境变量 `AGENT_HOME` 覆盖（测试与多环境隔离用）。
- **一个会话一个 JSON 文件**，不用 SQLite：数量少、可读、可手工删，
  也避免给部署环境多加一个数据库文件。
- **只存问答文本与生成的语句**，不存 DataFrame：结果集可能很大，
  而且下次续聊时重跑一遍更可靠（数据会变）。
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

HOME_ENV = "AGENT_HOME"
DEFAULT_HOME = "~/.data-agent"
MAX_NAME_LEN = 40
SAFE_NAME = re.compile(r"^[\w\-\u4e00-\u9fa5]{%d,%d}$" % (1, MAX_NAME_LEN))


class SessionError(Exception):
    """会话读写相关的错误（名字非法、文件损坏等）。"""


def home_dir() -> Path:
    """会话根目录：优先 `AGENT_HOME`，否则 `~/.data-agent`。"""
    return Path(os.path.expanduser(os.getenv(HOME_ENV) or DEFAULT_HOME))


def sessions_dir() -> Path:
    return home_dir() / "sessions"


def new_id() -> str:
    """按时间生成会话名，天然按时间排序，也方便人肉识别。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def check_name(name: str) -> str:
    """校验会话名，挡掉路径穿越（`../x`）这类输入。"""
    name = (name or "").strip()
    if not SAFE_NAME.match(name):
        raise SessionError(
            f"会话名不合法：{name!r}（1–{MAX_NAME_LEN} 个字母/数字/下划线/短横线/中文）"
        )
    return name


def path_of(name: str) -> Path:
    return sessions_dir() / f"{check_name(name)}.json"


def save_session(name: str, turns: List[dict], created: Optional[str] = None) -> dict:
    """整体写入一个会话文件。"""
    name = check_name(name)
    now = datetime.now().isoformat(timespec="seconds")
    session = {
        "name": name,
        "created": created or now,
        "updated": now,
        "turns": list(turns or []),
    }
    directory = sessions_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with open(path_of(name), "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return session


def load_session(name: Optional[str] = None) -> dict:
    """读取会话；name 为空时读最近一次（对应 `chat -c`）。"""
    if name:
        path = path_of(name)
    else:
        path = latest_path()
        if path is None:
            raise SessionError("还没有任何会话记录，先直接开一个交互会话吧")

    if not path.exists():
        raise SessionError(f"会话不存在：{path.stem}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SessionError(f"会话文件读取失败：{path}（{type(e).__name__}: {e}）") from e

    data.setdefault("name", path.stem)
    data.setdefault("turns", [])
    return data


def latest_path() -> Optional[Path]:
    """最近更新的会话文件；没有则返回 None。"""
    directory = sessions_dir()
    if not directory.exists():
        return None
    files = [p for p in directory.glob("*.json") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def append_turn(name: str, question: str, answer: str,
                route: str = "", statement: str = "") -> dict:
    """追加一轮问答（会话不存在就新建）。"""
    try:
        session = load_session(name)
    except SessionError:
        session = {"name": check_name(name), "created": None, "turns": []}

    session["turns"].append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer or "",
        "route": route or "",
        "statement": statement or "",
    })
    return save_session(session["name"], session["turns"], created=session.get("created"))


def list_sessions() -> List[dict]:
    """会话摘要列表，按更新时间倒序。"""
    directory = sessions_dir()
    if not directory.exists():
        return []

    summaries = []
    for path in directory.glob("*.json"):
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            # 单个文件坏了不该让整个列表挂掉
            summaries.append({"name": path.stem, "updated": "", "turns": 0,
                              "last": "（文件损坏，无法读取）"})
            continue
        turns = data.get("turns") or []
        summaries.append({
            "name": data.get("name") or path.stem,
            "updated": data.get("updated") or "",
            "turns": len(turns),
            "last": (turns[-1].get("question", "") if turns else ""),
        })
    return sorted(summaries, key=lambda s: s.get("updated") or "", reverse=True)


def turns_to_history(turns: List[dict], max_turns: int = 4) -> List[dict]:
    """把会话轮次还原成模型要的多轮上下文（只带文字，不带表格）。"""
    history: List[dict] = []
    for turn in turns[-max_turns:]:
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if question:
            history.append({"role": "user", "content": question})
        if answer:
            history.append({"role": "assistant", "content": answer})
    return history
