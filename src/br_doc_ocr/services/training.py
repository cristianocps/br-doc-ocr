"""
Training service for VLM fine-tuning.

Implements LoRA-based fine-tuning with checkpoint support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from br_doc_ocr.exceptions import TrainingError
from br_doc_ocr.lib.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    seed: int = 42

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """
    Create LoRA configuration for efficient fine-tuning.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability.
        target_modules: Modules to apply LoRA to.

    Returns:
        LoraConfig object.
    """
    try:
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
    except ImportError:
        logger.warning("PEFT not available, returning mock config")

        @dataclass
        class MockLoraConfig:
            r: int = r
            lora_alpha: int = lora_alpha
            lora_dropout: float = lora_dropout
            target_modules: list[str] = field(default_factory=lambda: target_modules or [])

        return MockLoraConfig()


class MetricsCollector:
    """Collects training metrics for logging and visualization."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def log(self, step: int, **metrics: Any) -> None:
        """Log metrics for a training step."""
        entry = {"step": step, **metrics}
        self.history.append(entry)

    def get_history(self) -> list[dict[str, Any]]:
        """Get all logged metrics."""
        return self.history

    def get_latest(self) -> dict[str, Any] | None:
        """Get most recent metrics."""
        return self.history[-1] if self.history else None

    def save(self, path: Path | str) -> None:
        """Save metrics history to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.history, indent=2))


def log_training_metrics(metrics: dict[str, Any]) -> str:
    """
    Log training metrics (PII-safe).

    Args:
        metrics: Dict of metric values.

    Returns:
        Log message string.
    """
    # Only log numeric/safe metrics
    safe_keys = ["loss", "accuracy", "epoch", "step", "learning_rate", "grad_norm"]
    safe_metrics = {k: v for k, v in metrics.items() if k in safe_keys}

    log_parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in safe_metrics.items()]
    log_message = " | ".join(log_parts)

    logger.info(f"Training: {log_message}")
    return log_message


class TrainingPipeline:
    """Training pipeline for VLM fine-tuning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        output_dir: str = "./outputs",
        config: TrainingConfig | None = None,
        resume_from_checkpoint: str | None = None,
        save_steps: int = 500,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.config = config or TrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
        )
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_steps = save_steps
        self.metrics_collector = MetricsCollector()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self) -> None:
        """Load the base model and apply LoRA."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )

            # Apply LoRA
            lora_config = get_lora_config(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
            )

            try:
                from peft import get_peft_model
                self.model = get_peft_model(self.model, lora_config)
                logger.info("Applied LoRA configuration")
            except ImportError:
                logger.warning("PEFT not available, training without LoRA")

        except Exception as e:
            raise TrainingError(f"Failed to load model: {e}") from e

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Any | None = None,
    ) -> dict[str, Any]:
        """
        Run training.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.

        Returns:
            Training metrics.
        """
        try:
            from transformers import Trainer, TrainingArguments

            if self.model is None:
                self.load_model()

            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                save_steps=self.save_steps,
                eval_steps=self.config.eval_steps,
                logging_steps=self.config.logging_steps,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                max_grad_norm=self.config.max_grad_norm,
                seed=self.config.seed,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_total_limit=3,
                load_best_model_at_end=bool(eval_dataset),
                report_to=[],  # Disable external reporting
            )

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )

            logger.info("Starting training...")

            if self.resume_from_checkpoint:
                train_result = self.trainer.train(
                    resume_from_checkpoint=self.resume_from_checkpoint
                )
            else:
                train_result = self.trainer.train()

            # Save final model
            self.trainer.save_model(str(self.output_dir / "final"))

            return train_result.metrics

        except Exception as e:
            raise TrainingError(f"Training failed: {e}") from e

    def save(self, path: str | None = None) -> Path:
        """Save the trained model."""
        save_path = Path(path) if path else self.output_dir / "model"
        save_path.mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save_pretrained(str(save_path))
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(save_path))

        logger.info(f"Model saved to: {save_path}")
        return save_path


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path | str,
    step: int,
) -> Path:
    """
    Save a training checkpoint.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        output_dir: Output directory.
        step: Training step number.

    Returns:
        Path to checkpoint directory.
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(checkpoint_dir))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(checkpoint_dir))

    logger.info(f"Saved checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint(checkpoint_path: str) -> Any:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory.

    Returns:
        Loaded model.
    """
    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype="auto",
            device_map="auto",
        )

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return model

    except Exception as e:
        raise TrainingError(f"Failed to load checkpoint: {e}") from e
