"""部署配置的离线检查（对应 S5.4）：不启动服务、不联网。

覆盖：
- 配置读取优先级：环境变量 > .env > Streamlit secrets（云端部署靠这条）
- 容器文件齐全，且 .dockerignore 真的排除了 .env（避免把密钥打进镜像）
- Streamlit 配置能被正确解析
- 文档里承诺的部署入口（app/main.py）真实存在

注意：本机未安装 Docker，镜像构建本身没有实测；这里只做静态检查。
"""
import os
import sys
import tomllib
import unittest
from unittest import mock

import app.config as config
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestConfigSource(unittest.TestCase):
    """密钥读取顺序决定了云端能不能跑起来。"""

    def test_env_var_wins(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-from-env"}):
            self.assertEqual(config._read_setting("DEEPSEEK_API_KEY"), "sk-from-env")

    def test_falls_back_to_streamlit_secrets(self):
        fake = mock.MagicMock()
        fake.secrets.get.side_effect = lambda key, default="": (
            "sk-from-secrets" if key == "DEEPSEEK_API_KEY" else default
        )
        env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.dict(sys.modules, {"streamlit": fake}):
                self.assertEqual(config._read_setting("DEEPSEEK_API_KEY"), "sk-from-secrets")

    def test_default_when_nothing_set(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_TEST_MISSING"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.dict(sys.modules, {"streamlit": None}):
                self.assertEqual(config._read_setting("AGENT_TEST_MISSING", "fallback"),
                                 "fallback")

    def test_limits_are_positive(self):
        for name in ("MAX_QUESTION_CHARS", "MAX_CODE_NODES", "EXEC_TIMEOUT_SECONDS",
                     "SQL_TIMEOUT_SECONDS", "MAX_RESULT_ROWS"):
            self.assertGreater(getattr(config, name), 0, f"{name} 必须为正数")


class TestConfigGate(unittest.TestCase):
    """check_config 是界面的总闸：没 Key 就停下并提示，而不是让用户问到一半才报错。"""

    def test_no_key_blocks_ui(self):
        with mock.patch.object(config, "LLM_API_KEY", ""):
            self.assertFalse(config.check_config())

    def test_placeholder_key_blocks_ui(self):
        """`.env.example` 里的占位值不该被当成有效 Key。"""
        with mock.patch.object(config, "LLM_API_KEY", "sk-在这里填你的key"):
            self.assertFalse(config.check_config())

    def test_real_key_passes(self):
        with mock.patch.object(config, "LLM_API_KEY", "sk-offline-placeholder"):
            self.assertTrue(config.check_config())


class TestContainerFiles(unittest.TestCase):
    def test_entrypoint_exists(self):
        self.assertTrue((ROOT / "app" / "main.py").exists())

    def test_dockerfile_key_instructions(self):
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM python:", text)
        self.assertIn("streamlit", text)
        self.assertIn("HEALTHCHECK", text)
        self.assertIn("EXPOSE 8501", text)
        # 用锁定文件安装，保证构建可复现
        self.assertIn("requirements.lock.txt", text)

    def test_dockerignore_excludes_secrets(self):
        lines = {line.strip() for line in
                 (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
        self.assertIn(".env", lines, "必须排除 .env，否则密钥会被打进镜像")
        self.assertIn(".venv/", lines)
        self.assertIn("docs/", lines)

    def test_gitignore_excludes_secrets_and_docs(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", text)
        self.assertIn("docs/", text)
        self.assertIn("output/", text)


class TestStreamlitConfig(unittest.TestCase):
    def test_config_parses(self):
        path = ROOT / ".streamlit" / "config.toml"
        self.assertTrue(path.exists())
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(data["server"]["headless"])
        self.assertEqual(data["server"]["maxUploadSize"], 20)


if __name__ == "__main__":
    unittest.main()
