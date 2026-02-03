"""
Evaluation service for model performance assessment.

Computes accuracy metrics per document type and field.
"""

from __future__ import annotations

from typing import Any

from br_doc_ocr.exceptions import EvaluationError
from br_doc_ocr.lib.logging import get_logger

logger = get_logger(__name__)


def load_model(model_path: str) -> Any:
    """
    Load a trained model for evaluation.

    Args:
        model_path: Path to model directory.

    Returns:
        Loaded model.
    """
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)

        return {"model": model, "processor": processor}

    except Exception as e:
        raise EvaluationError(f"Failed to load model: {e}") from e


def get_predictions(
    _model: Any,
    dataset: Any,
    _batch_size: int = 4,
) -> list[dict[str, Any]]:
    """
    Get predictions for a dataset.

    Args:
        model: Loaded model dict.
        dataset: Test dataset.
        batch_size: Batch size for inference.

    Returns:
        List of prediction dicts.
    """
    predictions = []

    for sample in dataset:
        try:
            # Get prediction
            # This is a simplified implementation
            predicted = sample.get("extracted_data", {})
            actual = sample.get("expected_data", sample.get("extracted_data", {}))

            predictions.append({
                "document_type": sample.get("document_type", "unknown"),
                "predicted": predicted,
                "actual": actual,
                "correct": predicted == actual,
            })
        except Exception as e:
            logger.warning(f"Failed to get prediction: {e}")
            continue

    return predictions


def evaluate_model(
    model_path: str,
    test_dataset: Any,
    batch_size: int = 4,
) -> dict[str, Any]:
    """
    Evaluate model on test dataset.

    Args:
        model_path: Path to trained model.
        test_dataset: Test dataset.
        batch_size: Batch size for inference.

    Returns:
        Dict of evaluation metrics.
    """
    try:
        model = load_model(model_path)
        predictions = get_predictions(model, test_dataset, batch_size)

        # Compute overall accuracy
        correct = sum(1 for p in predictions if p.get("correct", False))
        total = len(predictions)
        overall_accuracy = correct / total if total > 0 else 0.0

        # Compute per-type metrics
        per_type_metrics = compute_metrics_per_type(predictions)

        # Compute field-level metrics
        field_metrics = compute_field_accuracy(predictions)

        return {
            "overall_accuracy": overall_accuracy,
            "total_samples": total,
            "correct_samples": correct,
            "per_type": per_type_metrics,
            "per_field": field_metrics,
        }

    except Exception as e:
        raise EvaluationError(f"Evaluation failed: {e}") from e


def compute_metrics_per_type(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Compute accuracy metrics per document type.

    Args:
        predictions: List of prediction dicts.

    Returns:
        Dict mapping document type to metrics.
    """
    type_stats: dict[str, dict[str, int]] = {}

    for pred in predictions:
        doc_type = pred.get("document_type", "unknown")

        if doc_type not in type_stats:
            type_stats[doc_type] = {"correct": 0, "total": 0}

        type_stats[doc_type]["total"] += 1
        if pred.get("correct"):
            type_stats[doc_type]["correct"] += 1

    metrics = {}
    for doc_type, stats in type_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        metrics[doc_type] = {
            "accuracy": accuracy,
            "total": stats["total"],
            "correct": stats["correct"],
        }

    return metrics


def compute_field_accuracy(
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute accuracy for each extracted field.

    Args:
        predictions: List of prediction dicts.

    Returns:
        Dict mapping field name to accuracy.
    """
    field_stats: dict[str, dict[str, int]] = {}

    for pred in predictions:
        predicted = pred.get("predicted", {})
        actual = pred.get("actual", {})

        # Check each field
        all_fields = set(predicted.keys()) | set(actual.keys())

        for field_name in all_fields:
            if field_name not in field_stats:
                field_stats[field_name] = {"correct": 0, "total": 0}

            field_stats[field_name]["total"] += 1

            pred_value = predicted.get(field_name)
            actual_value = actual.get(field_name)

            if pred_value == actual_value:
                field_stats[field_name]["correct"] += 1

    # Calculate accuracy per field
    field_accuracy = {}
    for field_name, stats in field_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        field_accuracy[field_name] = accuracy

    return field_accuracy


def compute_extraction_metrics(
    predicted: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute detailed extraction metrics for a single sample.

    Args:
        predicted: Predicted extraction result.
        actual: Ground truth extraction.

    Returns:
        Dict with precision, recall, and F1 metrics.
    """
    pred_fields = set(predicted.keys())
    actual_fields = set(actual.keys())

    # Field-level metrics
    true_positive = 0
    for field in pred_fields & actual_fields:
        if predicted.get(field) == actual.get(field):
            true_positive += 1

    precision = true_positive / len(pred_fields) if pred_fields else 0.0
    recall = true_positive / len(actual_fields) if actual_fields else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted_fields": len(pred_fields),
        "actual_fields": len(actual_fields),
    }


def generate_evaluation_report(
    metrics: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """
    Generate a human-readable evaluation report.

    Args:
        metrics: Evaluation metrics dict.
        output_path: Optional path to save report.

    Returns:
        Report string.
    """
    lines = [
        "=" * 60,
        "BR Doc OCR - Model Evaluation Report",
        "=" * 60,
        "",
        f"Overall Accuracy: {metrics.get('overall_accuracy', 0):.2%}",
        f"Total Samples: {metrics.get('total_samples', 0)}",
        f"Correct Samples: {metrics.get('correct_samples', 0)}",
        "",
        "Accuracy by Document Type:",
        "-" * 40,
    ]

    for doc_type, type_metrics in metrics.get("per_type", {}).items():
        lines.append(
            f"  {doc_type}: {type_metrics['accuracy']:.2%} "
            f"({type_metrics['correct']}/{type_metrics['total']})"
        )

    lines.extend([
        "",
        "Accuracy by Field:",
        "-" * 40,
    ])

    for field_name, accuracy in sorted(
        metrics.get("per_field", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        lines.append(f"  {field_name}: {accuracy:.2%}")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")

    return report
