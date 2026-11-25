"""测试环境隔离。

`python -m unittest discover -s tests` 把 `tests/` 当作顶层目录，模块名是
`test_xxx` 而不是 `tests.test_xxx` —— 也就是说 **`tests/__init__.py` 不会被导入**。
隔离逻辑放那儿等于没写（踩过一次：环境变量设了，用例照样互相污染）。

所以单独放一个模块，需要跑完整问答的测试文件在顶部 `import _env` 即可。

两件事：
1. `AGENT_HOME` 指到一次性临时目录 —— 不碰用户真实的 `~/.data-agent`；
2. `AGENT_CACHE=0` 默认关掉 SQL 复用 —— 同一进程里前一个用例写入的缓存
   会被后一个用例命中，于是"应该调用生成"的断言莫名其妙失败。
   那是测试互相污染，不是功能坏了。

要测复用本身的用例（`test_sql_cache`）显式传 `use_cache=True`，并自己再隔离
一个 AGENT_HOME：它需要真实的读写行为，不能被关掉。
"""
import os
import tempfile

os.environ.setdefault("AGENT_HOME", tempfile.mkdtemp(prefix="data-agent-tests-"))
os.environ["AGENT_CACHE"] = "0"
