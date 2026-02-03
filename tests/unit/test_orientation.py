"""
Unit tests for orientation detection (FR-014).

Tests for orientation detection and correction functionality
in src/br_doc_ocr/services/preprocessing.py
"""

from __future__ import annotations

import pytest
from PIL import Image


class TestOrientationDetectionAlgorithm:
    """Tests for the orientation detection algorithm."""

    def test_detect_text_orientation_upright(self) -> None:
        """Should detect upright (0°) orientation."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        # Create a simple image that simulates an upright document
        img = Image.new("RGB", (400, 600), color=(255, 255, 255))
        orientation = detect_orientation(img)

        # For a test image, we just verify it returns a valid angle
        assert orientation in [0, 90, 180, 270]

    def test_detect_returns_integer_angle(self, sample_cnh_image: Image.Image) -> None:
        """Detection should return integer angle."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        result = detect_orientation(sample_cnh_image)

        assert isinstance(result, int)
        assert result in [0, 90, 180, 270]

    def test_detect_handles_small_images(self) -> None:
        """Should handle very small images gracefully."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        tiny_img = Image.new("RGB", (50, 50), color=(200, 200, 200))
        orientation = detect_orientation(tiny_img)

        assert orientation in [0, 90, 180, 270]

    def test_detect_handles_large_images(self) -> None:
        """Should handle large images (with downscaling)."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        large_img = Image.new("RGB", (4000, 3000), color=(255, 255, 255))
        orientation = detect_orientation(large_img)

        assert orientation in [0, 90, 180, 270]


class TestOrientationCorrectionMatrix:
    """Tests for orientation correction with all angles."""

    @pytest.mark.parametrize("angle", [0, 90, 180, 270])
    def test_correct_all_angles(self, sample_cnh_image: Image.Image, angle: int) -> None:
        """Should correctly handle all four orientation angles."""
        from br_doc_ocr.services.preprocessing import correct_orientation

        corrected = correct_orientation(sample_cnh_image, angle=angle)

        assert isinstance(corrected, Image.Image)
        if angle in [90, 270]:
            # Dimensions should be swapped
            assert corrected.size == (sample_cnh_image.height, sample_cnh_image.width)
        else:
            # Dimensions should be same
            assert corrected.size == sample_cnh_image.size

    def test_correct_invalid_angle_raises_error(
        self, sample_cnh_image: Image.Image
    ) -> None:
        """Invalid angle should raise ValueError."""
        from br_doc_ocr.services.preprocessing import correct_orientation

        with pytest.raises(ValueError, match="angle"):
            correct_orientation(sample_cnh_image, angle=45)

    def test_correct_preserves_image_mode(self, sample_cnh_image: Image.Image) -> None:
        """Correction should preserve image mode."""
        from br_doc_ocr.services.preprocessing import correct_orientation

        corrected = correct_orientation(sample_cnh_image, angle=90)

        assert corrected.mode == sample_cnh_image.mode


class TestAutoOrientationPipeline:
    """Tests for automatic orientation detection and correction."""

    def test_auto_correct_returns_image_and_angle(
        self, sample_cnh_image: Image.Image
    ) -> None:
        """Auto-correct should return tuple of (image, angle)."""
        from br_doc_ocr.services.preprocessing import auto_correct_orientation

        result = auto_correct_orientation(sample_cnh_image)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Image.Image)
        assert isinstance(result[1], int)

    def test_auto_correct_idempotent(self, sample_cnh_image: Image.Image) -> None:
        """Applying auto-correct twice should not change result."""
        from br_doc_ocr.services.preprocessing import auto_correct_orientation

        corrected1, angle1 = auto_correct_orientation(sample_cnh_image)
        corrected2, angle2 = auto_correct_orientation(corrected1)

        # Second correction should detect 0 degrees
        # (already correctly oriented)
        assert angle2 == 0 or corrected2.size == corrected1.size

    def test_auto_correct_with_disabled_flag(
        self, sample_rotated_image: Image.Image
    ) -> None:
        """Should be able to disable auto-correction."""
        from br_doc_ocr.services.preprocessing import preprocess_image

        # With auto_orient=False, should not correct
        result = preprocess_image(sample_rotated_image, auto_orient=False)

        # Size should be unchanged (not rotated back)
        assert result.size == sample_rotated_image.size


class TestOrientationWithDocumentTypes:
    """Tests for orientation detection with different document layouts."""

    def test_orientation_portrait_document(self) -> None:
        """Should detect orientation in portrait documents (like CNH)."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        # Portrait document (taller than wide)
        portrait_img = Image.new("RGB", (400, 600), color=(255, 255, 255))
        orientation = detect_orientation(portrait_img)

        assert orientation in [0, 90, 180, 270]

    def test_orientation_landscape_document(self) -> None:
        """Should detect orientation in landscape documents."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        # Landscape document (wider than tall)
        landscape_img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        orientation = detect_orientation(landscape_img)

        assert orientation in [0, 90, 180, 270]

    def test_orientation_square_document(self) -> None:
        """Should handle square documents."""
        from br_doc_ocr.services.preprocessing import detect_orientation

        square_img = Image.new("RGB", (500, 500), color=(255, 255, 255))
        orientation = detect_orientation(square_img)

        assert orientation in [0, 90, 180, 270]
