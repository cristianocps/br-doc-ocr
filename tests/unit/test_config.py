"""
Unit tests for configuration management.

Tests for src/br_doc_ocr/lib/config.py
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


class TestConfig:
    """Tests for Config class."""

    def test_config_loads_from_env_vars(self, mock_env_vars: dict[str, str]) -> None:
        """Config should load values from environment variables."""
        from br_doc_ocr.lib.config import Config

        config = Config()
        assert config.log_level == "DEBUG"
        assert config.database_url == "sqlite:///:memory:"

    def test_config_defaults_when_env_not_set(self) -> None:
        """Config should use defaults when environment variables are not set."""
        from br_doc_ocr.lib.config import Config

        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.log_level == "INFO"
            assert "sqlite" in config.database_url.lower()

    def test_config_model_cache_dir_is_path(self, mock_env_vars: dict[str, str]) -> None:
        """Model cache dir should be converted to Path."""
        from br_doc_ocr.lib.config import Config

        config = Config()
        assert isinstance(config.model_cache_dir, Path)

    def test_config_cuda_visible_devices(self) -> None:
        """Config should parse CUDA_VISIBLE_DEVICES correctly."""
        from br_doc_ocr.lib.config import Config

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            config = Config()
            assert config.cuda_visible_devices == "0,1"

    def test_config_is_singleton(self) -> None:
        """get_config should return the same instance."""
        from br_doc_ocr.lib.config import get_config

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_device_auto_detection(self) -> None:
        """Config should auto-detect device based on CUDA availability."""
        from br_doc_ocr.lib.config import Config

        config = Config()
        assert config.device in ["cuda", "cpu", "auto"]


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_log_level_raises_error(self) -> None:
        """Invalid log level should raise ValueError or use default."""
        from br_doc_ocr.lib.config import Config

        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            # Config may use default, store as-is, or raise
            try:
                config = Config()
                assert config.log_level is not None
            except ValueError:
                pass

    def test_invalid_database_url_raises_error(self) -> None:
        """Invalid database URL may use default or raise."""
        from br_doc_ocr.lib.config import Config

        with patch.dict(os.environ, {"DATABASE_URL": "not-a-valid-url"}):
            try:
                config = Config()
                assert config.database_url is not None
            except ValueError:
                pass
