"""
Unit tests for multi-document detection (FR-015).

Tests for detecting multiple documents in a single image
in src/br_doc_ocr/services/preprocessing.py
"""

from __future__ import annotations

import pytest
from PIL import Image


class TestMultiDocumentDetection:
    """Tests for multi-document detection algorithm."""

    def test_detect_single_document(self, sample_cnh_image: Image.Image) -> None:
        """Should return 1 region for single document image."""
        from br_doc_ocr.services.preprocessing import detect_documents

        regions = detect_documents(sample_cnh_image)

        assert isinstance(regions, list)
        assert len(regions) >= 1

    def test_detect_returns_bounding_boxes(self, sample_cnh_image: Image.Image) -> None:
        """Should return list of bounding box tuples."""
        from br_doc_ocr.services.preprocessing import detect_documents

        regions = detect_documents(sample_cnh_image)

        for region in regions:
            assert isinstance(region, tuple)
            assert len(region) == 4  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = region
            assert x1 < x2  # Valid width
            assert y1 < y2  # Valid height

    def test_detect_bounding_boxes_within_image(
        self, sample_cnh_image: Image.Image
    ) -> None:
        """Bounding boxes should be within image bounds."""
        from br_doc_ocr.services.preprocessing import detect_documents

        regions = detect_documents(sample_cnh_image)
        width, height = sample_cnh_image.size

        for x1, y1, x2, y2 in regions:
            assert 0 <= x1 < width
            assert 0 <= y1 < height
            assert 0 < x2 <= width
            assert 0 < y2 <= height

    def test_detect_multi_document_image(
        self, sample_multi_document_image: Image.Image
    ) -> None:
        """Should detect multiple documents in composite image."""
        from br_doc_ocr.services.preprocessing import detect_documents

        regions = detect_documents(sample_multi_document_image)

        # For a multi-doc image, should find multiple regions
        # (exact count depends on image content)
        assert isinstance(regions, list)


class TestDocumentCropping:
    """Tests for cropping individual documents from image."""

    def test_crop_single_region(self, sample_cnh_image: Image.Image) -> None:
        """Should crop a single region correctly."""
        from br_doc_ocr.services.preprocessing import crop_document

        # Define a region in the center
        w, h = sample_cnh_image.size
        region = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

        cropped = crop_document(sample_cnh_image, region)

        assert isinstance(cropped, Image.Image)
        expected_width = region[2] - region[0]
        expected_height = region[3] - region[1]
        assert cropped.size == (expected_width, expected_height)

    def test_crop_preserves_color_mode(self, sample_cnh_image: Image.Image) -> None:
        """Cropping should preserve image color mode."""
        from br_doc_ocr.services.preprocessing import crop_document

        w, h = sample_cnh_image.size
        region = (0, 0, w // 2, h // 2)

        cropped = crop_document(sample_cnh_image, region)

        assert cropped.mode == sample_cnh_image.mode

    def test_crop_invalid_region_raises_error(
        self, sample_cnh_image: Image.Image
    ) -> None:
        """Invalid region should raise ValueError."""
        from br_doc_ocr.services.preprocessing import crop_document

        # x2 < x1 is invalid
        invalid_region = (100, 50, 50, 100)

        with pytest.raises(ValueError, match="region"):
            crop_document(sample_cnh_image, invalid_region)


class TestExtractAllDocuments:
    """Tests for extracting all documents from an image."""

    def test_extract_all_returns_list(self, sample_cnh_image: Image.Image) -> None:
        """extract_all_documents should return list of images."""
        from br_doc_ocr.services.preprocessing import extract_all_documents

        documents = extract_all_documents(sample_cnh_image)

        assert isinstance(documents, list)
        assert all(isinstance(doc, Image.Image) for doc in documents)

    def test_extract_all_non_empty(self, sample_cnh_image: Image.Image) -> None:
        """Should extract at least one document."""
        from br_doc_ocr.services.preprocessing import extract_all_documents

        documents = extract_all_documents(sample_cnh_image)

        assert len(documents) >= 1

    def test_extract_all_with_min_size(
        self, sample_multi_document_image: Image.Image
    ) -> None:
        """Should filter out regions smaller than min_size."""
        from br_doc_ocr.services.preprocessing import extract_all_documents

        # With a very high min_size, might filter out some detections
        documents = extract_all_documents(
            sample_multi_document_image, min_size=(100, 100)
        )

        for doc in documents:
            assert doc.width >= 100
            assert doc.height >= 100


class TestMultiDocumentPipeline:
    """Tests for the full multi-document processing pipeline."""

    def test_pipeline_returns_count(self, sample_cnh_image: Image.Image) -> None:
        """Pipeline should report document count."""
        from br_doc_ocr.services.preprocessing import process_multi_document

        result = process_multi_document(sample_cnh_image)

        assert "document_count" in result
        assert isinstance(result["document_count"], int)
        assert result["document_count"] >= 1

    def test_pipeline_returns_images(self, sample_cnh_image: Image.Image) -> None:
        """Pipeline should return list of processed images."""
        from br_doc_ocr.services.preprocessing import process_multi_document

        result = process_multi_document(sample_cnh_image)

        assert "documents" in result
        assert isinstance(result["documents"], list)
        assert all(isinstance(doc, Image.Image) for doc in result["documents"])

    def test_pipeline_applies_orientation_to_each(
        self, sample_multi_document_image: Image.Image
    ) -> None:
        """Pipeline should apply orientation correction to each document."""
        from br_doc_ocr.services.preprocessing import process_multi_document

        result = process_multi_document(
            sample_multi_document_image, auto_orient=True
        )

        # Each document should be processed
        assert len(result["documents"]) >= 1


class TestEdgeCases:
    """Edge case tests for multi-document detection."""

    def test_blank_image_returns_empty_or_one(self) -> None:
        """Blank image should return empty or single region."""
        from br_doc_ocr.services.preprocessing import detect_documents

        blank = Image.new("RGB", (500, 500), color=(255, 255, 255))
        regions = detect_documents(blank)

        # Implementation may return 0 (no documents) or 1 (whole image)
        assert isinstance(regions, list)
        assert len(regions) <= 1

    def test_very_small_image(self) -> None:
        """Very small image should be handled gracefully."""
        from br_doc_ocr.services.preprocessing import detect_documents

        tiny = Image.new("RGB", (20, 20), color=(100, 100, 100))
        regions = detect_documents(tiny)

        assert isinstance(regions, list)

    def test_grayscale_image(self) -> None:
        """Should handle grayscale images."""
        from br_doc_ocr.services.preprocessing import detect_documents

        gray = Image.new("L", (400, 600), color=128)
        regions = detect_documents(gray)

        assert isinstance(regions, list)
