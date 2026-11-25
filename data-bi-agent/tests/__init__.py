"""测试包。

隔离逻辑在 `_env.py` —— `discover -s tests` 不会导入本文件（顶层目录是 tests），
所以只在以包方式（`-t .`）运行测试时这里才起作用，两种情况都覆盖到。
"""
from . import _env  # noqa: F401
