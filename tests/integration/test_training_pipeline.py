"""
Integration tests for the training pipeline.

Tests the full training flow from dataset to fine-tuned model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTrainingPipeline:
    """Integration tests for the training pipeline (US5)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_pipeline_initializes(self) -> None:
        """Training pipeline should initialize with config."""
        from br_doc_ocr.services.training import TrainingPipeline

        with patch("transformers.AutoModelForVision2Seq", create=True):
            pipeline = TrainingPipeline(
                model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                output_dir="/tmp/test_output",
            )

            assert pipeline is not None
            assert pipeline.output_dir == Path("/tmp/test_output")

    @pytest.mark.integration
    def test_training_with_lora_config(self) -> None:
        """Training should use LoRA for efficient fine-tuning."""
        from br_doc_ocr.services.training import get_lora_config

        lora_config = get_lora_config(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
        )

        assert lora_config is not None
        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32

    @pytest.mark.integration
    def test_training_saves_checkpoints(self, tmp_path: Path) -> None:
        """Training should save checkpoints at intervals."""
        from br_doc_ocr.services.training import TrainingPipeline

        with patch("transformers.AutoModelForVision2Seq", create=True):
            with patch("br_doc_ocr.services.training.Trainer") as mock_trainer:
                mock_trainer_instance = MagicMock()
                mock_trainer.return_value = mock_trainer_instance

                pipeline = TrainingPipeline(
                    model_name="test-model",
                    output_dir=str(tmp_path),
                    save_steps=100,
                )

                # Verify checkpoint config
                assert pipeline.save_steps == 100

    @pytest.mark.integration
    def test_training_can_resume_from_checkpoint(self, tmp_path: Path) -> None:
        """Training should resume from checkpoint."""
        from br_doc_ocr.services.training import TrainingPipeline

        checkpoint_dir = tmp_path / "checkpoint-100"
        checkpoint_dir.mkdir()

        with patch("transformers.AutoModelForVision2Seq", create=True):
            pipeline = TrainingPipeline(
                model_name="test-model",
                output_dir=str(tmp_path),
                resume_from_checkpoint=str(checkpoint_dir),
            )

            assert pipeline.resume_from_checkpoint == str(checkpoint_dir)


class TestEvaluationPipeline:
    """Integration tests for model evaluation."""

    @pytest.mark.integration
    def test_evaluation_computes_accuracy(self) -> None:
        """Evaluation should compute accuracy per document type."""
        from br_doc_ocr.services.evaluation import evaluate_model

        with patch("br_doc_ocr.services.evaluation.load_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            # Mock predictions (must include "correct" key)
            predictions = [
                {"document_type": "cnh", "predicted": {"cpf": "123"}, "actual": {"cpf": "123"}, "correct": True},
                {"document_type": "cnh", "predicted": {"cpf": "456"}, "actual": {"cpf": "456"}, "correct": True},
                {"document_type": "rg", "predicted": {"rg": "789"}, "actual": {"rg": "789"}, "correct": True},
            ]

            with patch("br_doc_ocr.services.evaluation.get_predictions", return_value=predictions):
                metrics = evaluate_model(
                    model_path="/tmp/model",
                    test_dataset=MagicMock(),
                )

                assert "accuracy" in metrics or "overall_accuracy" in metrics

    @pytest.mark.integration
    def test_evaluation_per_document_type(self) -> None:
        """Evaluation should report metrics per document type."""
        from br_doc_ocr.services.evaluation import compute_metrics_per_type

        predictions = [
            {"document_type": "cnh", "correct": True},
            {"document_type": "cnh", "correct": True},
            {"document_type": "cnh", "correct": False},
            {"document_type": "rg", "correct": True},
            {"document_type": "rg", "correct": True},
        ]

        metrics = compute_metrics_per_type(predictions)

        assert "cnh" in metrics
        assert "rg" in metrics
        assert metrics["cnh"]["accuracy"] == pytest.approx(2/3, rel=0.01)
        assert metrics["rg"]["accuracy"] == 1.0

    @pytest.mark.integration
    def test_evaluation_field_level_accuracy(self) -> None:
        """Evaluation should compute field-level accuracy."""
        from br_doc_ocr.services.evaluation import compute_field_accuracy

        predictions = [
            {
                "predicted": {"nome": "JOÃO", "cpf": "123"},
                "actual": {"nome": "JOÃO", "cpf": "123"},
            },
            {
                "predicted": {"nome": "MARIA", "cpf": "999"},
                "actual": {"nome": "MARIA", "cpf": "456"},
            },
        ]

        field_accuracy = compute_field_accuracy(predictions)

        assert field_accuracy["nome"] == 1.0  # Both correct
        assert field_accuracy["cpf"] == 0.5   # 1 of 2 correct


class TestTrainingCLI:
    """Integration tests for training CLI commands."""

    @pytest.mark.integration
    def test_train_cli_command(self, tmp_path: Path) -> None:
        """Train CLI command should work."""
        from typer.testing import CliRunner

        from br_doc_ocr.cli.main import app

        runner = CliRunner()

        result = runner.invoke(
            app,
            ["train", "--help"],
        )

        assert result.exit_code == 0
        assert "dataset" in result.stdout.lower() or "train" in result.stdout.lower()

    @pytest.mark.integration
    def test_evaluate_cli_command(self) -> None:
        """Evaluate CLI command should work."""
        from typer.testing import CliRunner

        from br_doc_ocr.cli.main import app

        runner = CliRunner()

        result = runner.invoke(
            app,
            ["evaluate", "--help"],
        )

        assert result.exit_code == 0
        assert "model" in result.stdout.lower() or "evaluate" in result.stdout.lower()


class TestMetricsLogging:
    """Tests for training metrics logging."""

    @pytest.mark.integration
    def test_metrics_logged_without_pii(self) -> None:
        """Metrics logging should not contain PII."""
        from br_doc_ocr.services.training import log_training_metrics

        metrics = {
            "loss": 0.5,
            "accuracy": 0.92,
            "epoch": 1,
            "step": 100,
        }

        # Should not raise and should not log PII
        log_message = log_training_metrics(metrics)

        # Verify no PII fields in log
        pii_fields = ["nome", "cpf", "rg", "data_nascimento"]
        for field in pii_fields:
            assert field not in log_message.lower()

    @pytest.mark.integration
    def test_loss_curve_data_collected(self) -> None:
        """Training should collect loss curve data."""
        from br_doc_ocr.services.training import MetricsCollector

        collector = MetricsCollector()

        collector.log(step=1, loss=1.0)
        collector.log(step=2, loss=0.8)
        collector.log(step=3, loss=0.6)

        history = collector.get_history()

        assert len(history) == 3
        assert history[0]["loss"] == 1.0
        assert history[-1]["loss"] == 0.6


class TestModelCheckpointing:
    """Tests for model checkpointing."""

    @pytest.mark.integration
    def test_save_checkpoint(self, tmp_path: Path) -> None:
        """Should save model checkpoint."""
        from br_doc_ocr.services.training import save_checkpoint

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        checkpoint_path = save_checkpoint(
            model=mock_model,
            tokenizer=mock_tokenizer,
            output_dir=tmp_path,
            step=100,
        )

        assert checkpoint_path is not None
        assert "checkpoint" in str(checkpoint_path)

    @pytest.mark.integration
    def test_load_checkpoint(self, tmp_path: Path) -> None:
        """Should load model from checkpoint."""
        from br_doc_ocr.services.training import load_checkpoint

        checkpoint_dir = tmp_path / "checkpoint-100"
        checkpoint_dir.mkdir()

        with patch("transformers.AutoModelForVision2Seq", create=True) as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            model = load_checkpoint(str(checkpoint_dir))

            assert model is not None
