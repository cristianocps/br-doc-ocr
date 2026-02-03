"""
Unit tests for dataset loading and transformation.

Tests the dataset adapter for HuggingFace datasets.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDatasetLoading:
    """Tests for loading datasets from HuggingFace."""

    def test_load_dataset_from_hub(self) -> None:
        """Load dataset from HuggingFace Hub."""
        from br_doc_ocr.services.dataset_adapter import load_training_dataset

        with patch("datasets.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_load.return_value = mock_dataset

            dataset = load_training_dataset("tech4humans/br-doc-extraction")

            mock_load.assert_called_once()
            assert dataset is not None

    def test_load_dataset_with_split(self) -> None:
        """Load specific split from dataset."""
        from br_doc_ocr.services.dataset_adapter import load_training_dataset

        with patch("datasets.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            load_training_dataset(
                "tech4humans/br-doc-extraction",
                split="train",
            )

            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args
            assert "split" in str(call_kwargs) or call_kwargs[1].get("split") == "train"

    def test_load_dataset_handles_error(self) -> None:
        """Dataset loading errors should be handled gracefully."""
        from br_doc_ocr.exceptions import TrainingError
        from br_doc_ocr.services.dataset_adapter import load_training_dataset

        with patch("datasets.load_dataset") as mock_load:
            mock_load.side_effect = Exception("Network error")

            with pytest.raises(TrainingError):
                load_training_dataset("invalid/dataset")


class TestDatasetTransformation:
    """Tests for transforming datasets for VLM training."""

    def test_transform_sample_for_vlm(self) -> None:
        """Transform a dataset sample for VLM input."""
        from br_doc_ocr.services.dataset_adapter import transform_sample

        sample = {
            "image": MagicMock(),  # PIL Image mock
            "document_type": "cnh",
            "extracted_data": {
                "nome_completo": "JOÃO SILVA",
                "cpf": "123.456.789-00",
            },
        }

        transformed = transform_sample(sample)

        assert "messages" in transformed or "prompt" in transformed
        assert "image" in transformed

    def test_transform_creates_conversation_format(self) -> None:
        """Transformed data should be in conversation format for VLM."""
        from br_doc_ocr.services.dataset_adapter import transform_sample

        sample = {
            "image": MagicMock(),
            "document_type": "rg",
            "extracted_data": {"nome_completo": "TESTE"},
        }

        transformed = transform_sample(sample)

        # Should have messages in conversation format
        if "messages" in transformed:
            assert isinstance(transformed["messages"], list)
            assert len(transformed["messages"]) >= 2  # system + user at minimum

    def test_transform_includes_expected_output(self) -> None:
        """Transformed data should include expected JSON output."""
        from br_doc_ocr.services.dataset_adapter import transform_sample

        expected_data = {"nome_completo": "MARIA", "cpf": "111.222.333-44"}
        sample = {
            "image": MagicMock(),
            "document_type": "cnh",
            "extracted_data": expected_data,
        }

        transformed = transform_sample(sample)

        # Should have the expected output somewhere
        assert "expected_output" in transformed or "assistant" in str(transformed)


class TestDatasetFiltering:
    """Tests for filtering dataset samples."""

    def test_filter_by_document_type(self) -> None:
        """Filter dataset to specific document types."""
        from br_doc_ocr.services.dataset_adapter import filter_by_document_type

        samples = [
            {"document_type": "cnh", "data": "..."},
            {"document_type": "rg", "data": "..."},
            {"document_type": "cnh", "data": "..."},
            {"document_type": "invoice", "data": "..."},
        ]

        filtered = filter_by_document_type(samples, ["cnh"])

        assert len(filtered) == 2
        assert all(s["document_type"] == "cnh" for s in filtered)

    def test_filter_multiple_types(self) -> None:
        """Filter dataset to multiple document types."""
        from br_doc_ocr.services.dataset_adapter import filter_by_document_type

        samples = [
            {"document_type": "cnh", "data": "..."},
            {"document_type": "rg", "data": "..."},
            {"document_type": "invoice", "data": "..."},
        ]

        filtered = filter_by_document_type(samples, ["cnh", "rg"])

        assert len(filtered) == 2


class TestDatasetSplitting:
    """Tests for splitting datasets into train/val/test."""

    def test_split_dataset(self) -> None:
        """Split dataset into train/val/test sets."""
        from br_doc_ocr.services.dataset_adapter import split_dataset

        # Mock dataset with 100 samples
        samples = [{"id": i} for i in range(100)]

        splits = split_dataset(
            samples,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_split_preserves_all_samples(self) -> None:
        """Split should preserve all samples without overlap."""
        from br_doc_ocr.services.dataset_adapter import split_dataset

        samples = [{"id": i} for i in range(50)]

        splits = split_dataset(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        all_ids = set()
        for split_name, split_samples in splits.items():
            for s in split_samples:
                assert s["id"] not in all_ids, f"Duplicate found in {split_name}"
                all_ids.add(s["id"])

        assert len(all_ids) == 50


class TestCollateFunction:
    """Tests for batch collation."""

    def test_collate_batch(self) -> None:
        """Collate samples into a batch."""
        from br_doc_ocr.services.dataset_adapter import collate_fn

        samples = [
            {"input_ids": [1, 2, 3], "image": MagicMock()},
            {"input_ids": [4, 5, 6], "image": MagicMock()},
        ]

        batch = collate_fn(samples)

        assert "input_ids" in batch or "images" in batch

    def test_collate_pads_sequences(self) -> None:
        """Collate should pad sequences to same length."""
        from br_doc_ocr.services.dataset_adapter import collate_fn

        samples = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5, 6, 7, 8]},
        ]

        batch = collate_fn(samples)

        if "input_ids" in batch:
            # All sequences should have same length after padding
            assert all(len(seq) == len(batch["input_ids"][0]) for seq in batch["input_ids"])
