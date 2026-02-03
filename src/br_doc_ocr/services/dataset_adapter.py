"""
Dataset adapter for HuggingFace datasets.

Transforms datasets for VLM fine-tuning.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from typing import Any

from br_doc_ocr.exceptions import TrainingError
from br_doc_ocr.lib.logging import get_logger

logger = get_logger(__name__)


def load_training_dataset(
    dataset_name: str,
    split: str | None = None,
    cache_dir: str | None = None,
) -> Any:
    """
    Load a dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split ("train", "test", "validation").
        cache_dir: Directory to cache downloaded data.

    Returns:
        HuggingFace Dataset object.

    Raises:
        TrainingError: If dataset loading fails.
    """
    try:
        from datasets import load_dataset

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        if split:
            kwargs["split"] = split

        dataset = load_dataset(dataset_name, **kwargs)
        logger.info(f"Loaded dataset: {dataset_name}")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise TrainingError(f"Failed to load dataset: {e}") from e


def transform_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Transform a dataset sample for VLM training.

    Converts raw dataset sample to conversation format expected by the model.

    Args:
        sample: Raw dataset sample with image and extracted_data.

    Returns:
        Transformed sample with messages in conversation format.
    """
    document_type = sample.get("document_type", "document")
    extracted_data = sample.get("extracted_data", {})

    # Build conversation format
    system_message = (
        "You are an expert document extraction assistant. "
        "Extract structured data from the document image and return valid JSON."
    )

    user_message = (
        f"Extract all relevant information from this {document_type} document. "
        "Return the extracted data as a JSON object."
    )

    assistant_message = json.dumps(extracted_data, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    return {
        "messages": messages,
        "image": sample.get("image"),
        "expected_output": extracted_data,
        "document_type": document_type,
    }


def transform_dataset(
    dataset: Any,
    transform_fn: Callable[[dict], dict] | None = None,
) -> list[dict[str, Any]]:
    """
    Transform all samples in a dataset.

    Args:
        dataset: HuggingFace Dataset or list of samples.
        transform_fn: Custom transformation function.

    Returns:
        List of transformed samples.
    """
    fn = transform_fn or transform_sample

    transformed = []
    for sample in dataset:
        try:
            transformed.append(fn(sample))
        except Exception as e:
            logger.warning(f"Failed to transform sample: {e}")
            continue

    return transformed


def filter_by_document_type(
    samples: Sequence[dict[str, Any]],
    document_types: list[str],
) -> list[dict[str, Any]]:
    """
    Filter samples to specific document types.

    Args:
        samples: List of samples.
        document_types: List of document types to keep.

    Returns:
        Filtered list of samples.
    """
    return [s for s in samples if s.get("document_type") in document_types]


def split_dataset(
    samples: Sequence[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """
    Split dataset into train/val/test sets.

    Args:
        samples: List of samples to split.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.
        test_ratio: Ratio for test set.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' keys.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"

    samples_list = list(samples)
    random.seed(seed)
    random.shuffle(samples_list)

    n = len(samples_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": samples_list[:train_end],
        "val": samples_list[train_end:val_end],
        "test": samples_list[val_end:],
    }


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate samples into a batch.

    Args:
        samples: List of samples to collate.

    Returns:
        Batched data dict.
    """
    batch: dict[str, list] = {}

    for sample in samples:
        for key, value in sample.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(value)

    # Pad sequences if needed
    if "input_ids" in batch:
        max_len = max(len(seq) for seq in batch["input_ids"])
        padded = []
        for seq in batch["input_ids"]:
            padded.append(seq + [0] * (max_len - len(seq)))
        batch["input_ids"] = padded

    return batch


def create_dataloader(
    samples: list[dict[str, Any]],
    batch_size: int = 4,
    shuffle: bool = True,
) -> Any:
    """
    Create a DataLoader for training.

    Args:
        samples: List of training samples.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.

    Returns:
        PyTorch DataLoader.
    """
    try:
        from torch.utils.data import DataLoader

        return DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    except ImportError:
        logger.warning("PyTorch not available, returning samples directly")
        return samples
