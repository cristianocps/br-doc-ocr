"""
Integration tests for CLI commands.

Tests the CLI interface for extraction and classification.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from br_doc_ocr.cli.main import app

runner = CliRunner()


class TestExtractCLICommand:
    """Integration tests for 'br-doc-ocr extract' command (US1)."""

    @pytest.mark.integration
    def test_extract_command_with_image_path(
        self, temp_image_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Extract command should accept image path."""
        result = runner.invoke(app, ["extract", str(temp_image_path)])

        # Command should run without error (even if not fully implemented)
        assert result.exit_code == 0

    @pytest.mark.integration
    def test_extract_command_with_type_option(
        self, temp_image_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Extract command should accept --type option."""
        result = runner.invoke(
            app, ["extract", str(temp_image_path), "--type", "cnh"]
        )

        assert result.exit_code == 0

    @pytest.mark.integration
    def test_extract_command_with_confidence_flag(
        self, temp_image_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Extract command should accept --confidence flag."""
        result = runner.invoke(
            app, ["extract", str(temp_image_path), "--confidence"]
        )

        assert result.exit_code == 0

    @pytest.mark.integration
    def test_extract_command_with_output_file(
        self, temp_image_path: Path, tmp_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Extract command should accept --output option."""
        output_file = tmp_path / "result.json"

        result = runner.invoke(
            app, ["extract", str(temp_image_path), "--output", str(output_file)]
        )

        assert result.exit_code == 0

    @pytest.mark.integration
    def test_extract_command_with_schema_option(
        self, temp_image_path: Path, tmp_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Extract command should accept --schema option."""
        import json

        schema_file = tmp_path / "custom_schema.json"
        schema_file.write_text(json.dumps({
            "type": "object",
            "properties": {"field1": {"type": "string"}},
        }))

        result = runner.invoke(
            app, ["extract", str(temp_image_path), "--schema", str(schema_file)]
        )

        assert result.exit_code == 0

    @pytest.mark.integration
    def test_extract_nonexistent_file_shows_error(self) -> None:
        """Extract with nonexistent file should show error."""
        result = runner.invoke(app, ["extract", "/nonexistent/image.jpg"])

        # Should fail (exit 2 = file not found, 4 = extraction error)
        assert result.exit_code != 0
        assert result.exit_code in [1, 2, 4]


class TestClassifyCLICommand:
    """Integration tests for 'br-doc-ocr classify' command."""

    @pytest.mark.integration
    def test_classify_command_with_image_path(
        self, temp_image_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Classify command should accept image path."""
        result = runner.invoke(app, ["classify", str(temp_image_path)])

        assert result.exit_code == 0

    @pytest.mark.integration
    def test_classify_command_with_output_option(
        self, temp_image_path: Path, tmp_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Classify command should accept --output option."""
        output_file = tmp_path / "classification.json"

        result = runner.invoke(
            app, ["classify", str(temp_image_path), "--output", str(output_file)]
        )

        assert result.exit_code == 0


class TestVersionCommand:
    """Tests for 'br-doc-ocr version' command."""

    def test_version_command_shows_version(self) -> None:
        """Version command should display version number."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.stdout or "version" in result.stdout.lower()


class TestInfoCommand:
    """Tests for 'br-doc-ocr info' command."""

    def test_info_command_shows_configuration(self) -> None:
        """Info command should display configuration."""
        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "Device" in result.stdout or "Model" in result.stdout


class TestBatchCommand:
    """Tests for 'br-doc-ocr batch' command."""

    @pytest.mark.integration
    def test_batch_command_accepts_directory(
        self, tmp_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Batch command should accept input directory."""
        result = runner.invoke(app, ["batch", str(tmp_path)])

        assert result.exit_code == 0

    @pytest.mark.integration
    def test_batch_command_with_workers_option(
        self, tmp_path: Path, mock_vlm: MagicMock
    ) -> None:
        """Batch command should accept --workers option."""
        result = runner.invoke(
            app, ["batch", str(tmp_path), "--workers", "2"]
        )

        assert result.exit_code == 0
