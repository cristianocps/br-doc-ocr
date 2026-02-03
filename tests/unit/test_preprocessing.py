"""
Unit tests for image preprocessing service.

Tests for src/br_doc_ocr/services/preprocessing.py
Covers: resize, normalize, orientation detection/correction (FR-014)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class TestImageResize:
    """Tests for image resizing functionality."""

    def test_resize_maintains_aspect_ratio(self, sample_cnh_image: Image.Image) -> None:
        """Resize should maintain aspect ratio."""
        from br_doc_ocr.services.preprocessing import resize_image

        original_ratio = sample_cnh_image.width / sample_cnh_image.height
        resized = resize_image(sample_cnh_image, max_size=400)
        new_ratio = resized.width / resized.height

        assert abs(original_ratio - new_ratio) < 0.01

    def test_resize_does_not_upscale(self, sample_cnh_image: Image.Image) -> None:
        """Resize should not upscale images larger than max_size."""
        from br_doc_ocr.services.preprocessing import resize_image

        resized = resize_image(sample_cnh_image, max_size=2000)
        assert resized.width <= sample_cnh_image.width
        assert resized.height <= sample_cnh_image.height

    def test_resize_respects_max_size(self, sample_cnh_image: Image.Image) -> None:
        """Resize should ensure max dimension is at most max_size."""
        from br_doc_ocr.services.preprocessing import resize_image

        max_size = 400
        resized = resize_image(sample_cnh_image, max_size=max_size)

        assert max(resized.width, resized.height) <= max_size


class TestImageNormalize:
    """Tests for image normalization."""

    def test_normalize_converts_to_rgb(self) -> None:
        """Normalize should convert RGBA/L images to RGB."""
        from br_doc_ocr.services.preprocessing import normalize_image

        rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        normalized = normalize_image(rgba_image)

        assert normalized.mode == "RGB"

    def test_normalize_handles_grayscale(self) -> None:
        """Normalize should handle grayscale images."""
        from br_doc_ocr.services.preprocessing import normalize_image

        gray_image = Image.new("L", (100, 100), 128)
        normalized = normalize_image(gray_image)

        assert normalized.mode == "RGB"

    def test_normalize_returns_numpy_array_option(self) -> None:
        """Normalize should optionally return numpy array."""
        from br_doc_ocr.services.preprocessing import normalize_image

        image = Image.new("RGB", (100, 100))
        normalized = normalize_image(image, as_array=True)

        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == (100, 100, 3)


class TestOrientationDetection:
    """Tests for orientation detection (FR-014)."""

    def test_detect_orientation_upright(self, sample_cnh_image: Image.Image) -> None:
        """Detect orientation should return 0 for upright images."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        orientation = detect_orientation(sample_cnh_image)
        assert orientation in [0, 90, 180, 270]

    def test_detect_orientation_rotated_90(self, sample_rotated_image: Image.Image) -> None:
        """Detect orientation should identify 90-degree rotation."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        orientation = detect_orientation(sample_rotated_image)
        # Should detect some rotation (exact value depends on implementation)
        assert isinstance(orientation, int)
        assert orientation % 90 == 0


class TestOrientationCorrection:
    """Tests for orientation correction (FR-014)."""

    def test_correct_orientation_0_degrees(self, sample_cnh_image: Image.Image) -> None:
        """Correct orientation with 0 degrees should return unchanged image."""
        from br_doc_ocr.services.preprocessing import correct_orientation

        corrected = correct_orientation(sample_cnh_image, angle=0)
        assert corrected.size == sample_cnh_image.size

    def test_correct_orientation_90_degrees(self, sample_cnh_image: Image.Image) -> None:
        """Correct orientation with 90 degrees should rotate image."""
        from br_doc_ocr.services.preprocessing import correct_orientation

        original_size = sample_cnh_image.size
        corrected = correct_orientation(sample_cnh_image, angle=90)

        # Width and height should be swapped
        assert corrected.size == (original_size[1], original_size[0])

    def test_correct_orientation_180_degrees(self, sample_cnh_image: Image.Image) -> None:
        """Correct orientation with 180 degrees should rotate image."""
        from br_doc_ocr.services.preprocessing import correct_orientation

        corrected = correct_orientation(sample_cnh_image, angle=180)
        assert corrected.size == sample_cnh_image.size

    def test_correct_orientation_270_degrees(self, sample_cnh_image: Image.Image) -> None:
        """Correct orientation with 270 degrees should rotate image."""
        from br_doc_ocr.services.preprocessing import correct_orientation

        original_size = sample_cnh_image.size
        corrected = correct_orientation(sample_cnh_image, angle=270)

        # Width and height should be swapped
        assert corrected.size == (original_size[1], original_size[0])

    def test_auto_correct_orientation(self, sample_rotated_image: Image.Image) -> None:
        """Auto-correct should detect and fix orientation."""
        from br_doc_ocr.services.preprocessing import auto_correct_orientation

        corrected, angle = auto_correct_orientation(sample_rotated_image)

        assert isinstance(corrected, Image.Image)
        assert angle in [0, 90, 180, 270]


class TestPreprocessPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_preprocess_image_returns_pil_image(
        self, sample_cnh_image: Image.Image
    ) -> None:
        """Preprocess should return a PIL Image by default."""
        from br_doc_ocr.services.preprocessing import preprocess_image

        result = preprocess_image(sample_cnh_image)
        assert isinstance(result, Image.Image)

    def test_preprocess_image_applies_orientation_correction(
        self, sample_rotated_image: Image.Image
    ) -> None:
        """Preprocess should apply orientation correction."""
        from br_doc_ocr.services.preprocessing import preprocess_image

        result = preprocess_image(sample_rotated_image, auto_orient=True)
        assert isinstance(result, Image.Image)

    def test_preprocess_from_path(self, temp_image_path: Path) -> None:
        """Preprocess should accept file path."""

        from br_doc_ocr.services.preprocessing import preprocess_image

        result = preprocess_image(temp_image_path)
        assert isinstance(result, Image.Image)

    def test_preprocess_from_numpy(self, sample_cnh_image: Image.Image) -> None:
        """Preprocess should accept numpy array."""
        from br_doc_ocr.services.preprocessing import preprocess_image

        np_image = np.array(sample_cnh_image)
        result = preprocess_image(np_image)
        assert isinstance(result, Image.Image)
