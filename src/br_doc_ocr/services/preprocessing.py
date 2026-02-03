"""
Image preprocessing service.

Handles image loading, resizing, normalization, orientation detection/correction (FR-014),
and multi-document detection (FR-015).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

from br_doc_ocr.exceptions import ImageFormatError, ImageLoadError

# Type alias for image input
ImageInput = Union[str, Path, Image.Image, np.ndarray]

# Supported image formats
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


def load_image(source: ImageInput) -> Image.Image:
    """
    Load an image from various sources.

    Args:
        source: Path string, Path object, PIL Image, or numpy array.

    Returns:
        PIL Image object.

    Raises:
        ImageLoadError: If image cannot be loaded.
        ImageFormatError: If image format is not supported.
    """
    if isinstance(source, Image.Image):
        return source

    if isinstance(source, np.ndarray):
        return Image.fromarray(source)

    # Handle file path
    path = Path(source) if isinstance(source, str) else source

    if not path.exists():
        raise ImageLoadError(str(path), "File not found")

    try:
        img = Image.open(path)
        img.load()  # Force load to catch errors

        # Check format
        if img.format and img.format.upper() not in SUPPORTED_FORMATS:
            raise ImageFormatError(img.format)

        return img
    except ImageFormatError:
        raise
    except Exception as e:
        raise ImageLoadError(str(path), str(e)) from e


def resize_image(
    image: Image.Image,
    max_size: int = 1024,
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image to resize.
        max_size: Maximum dimension (width or height).
        resample: Resampling method.

    Returns:
        Resized PIL Image.
    """
    width, height = image.size

    # Don't upscale
    if max(width, height) <= max_size:
        return image

    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), resample=resample)


def normalize_image(
    image: Image.Image,
    as_array: bool = False,
) -> Image.Image | np.ndarray:
    """
    Normalize image to RGB format.

    Args:
        image: PIL Image to normalize.
        as_array: If True, return numpy array.

    Returns:
        Normalized PIL Image or numpy array.
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        if image.mode == "RGBA":
            # Handle transparency by compositing on white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")

    if as_array:
        return np.array(image)

    return image


def detect_orientation(image: Image.Image) -> int:
    """
    Detect document orientation.

    Uses heuristics based on text direction and document layout.

    Args:
        image: PIL Image to analyze.

    Returns:
        Detected rotation angle (0, 90, 180, or 270).
    """
    # For a robust implementation, this would use:
    # 1. Text detection and reading direction analysis
    # 2. Document layout analysis
    # 3. OCR confidence at different rotations

    # Simplified implementation for now:
    # Assume portrait documents are correctly oriented
    # Landscape documents might be rotated

    width, height = image.size

    # If significantly wider than tall, might be rotated 90°
    # This is a placeholder - real implementation would be more sophisticated
    if width > height * 1.5:
        # Could be 90 or 270, would need more analysis
        return 0  # Default to no rotation for safety

    return 0


def correct_orientation(
    image: Image.Image,
    angle: int,
) -> Image.Image:
    """
    Correct image orientation by rotating.

    Args:
        image: PIL Image to rotate.
        angle: Rotation angle (must be 0, 90, 180, or 270).

    Returns:
        Rotated PIL Image.

    Raises:
        ValueError: If angle is not valid.
    """
    if angle not in [0, 90, 180, 270]:
        raise ValueError(f"Invalid angle: {angle}. Must be 0, 90, 180, or 270.")

    if angle == 0:
        return image

    # PIL rotates counter-clockwise, so we negate
    # expand=True ensures the full rotated image fits
    return image.rotate(-angle, expand=True)


def auto_correct_orientation(image: Image.Image) -> tuple[Image.Image, int]:
    """
    Automatically detect and correct orientation.

    Args:
        image: PIL Image to process.

    Returns:
        Tuple of (corrected image, detected angle).
    """
    angle = detect_orientation(image)
    corrected = correct_orientation(image, angle)
    return corrected, angle


def detect_documents(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Detect document regions in an image.

    Implements FR-015: multi-document detection.

    Args:
        image: PIL Image to analyze.

    Returns:
        List of bounding boxes as (x1, y1, x2, y2) tuples.
    """
    # For a full implementation, this would use:
    # 1. Edge detection
    # 2. Contour finding
    # 3. Rectangle detection
    # 4. Document boundary heuristics

    # Simplified implementation: return the full image as a single document
    width, height = image.size

    # Basic heuristic: if image is very wide, might contain multiple side-by-side docs
    # This is a placeholder for more sophisticated detection
    if width > height * 2:
        # Potentially two documents side by side
        mid_x = width // 2
        return [
            (0, 0, mid_x, height),
            (mid_x, 0, width, height),
        ]

    # Single document covering the whole image
    return [(0, 0, width, height)]


def crop_document(
    image: Image.Image,
    region: tuple[int, int, int, int],
) -> Image.Image:
    """
    Crop a document region from an image.

    Args:
        image: PIL Image to crop.
        region: Bounding box as (x1, y1, x2, y2).

    Returns:
        Cropped PIL Image.

    Raises:
        ValueError: If region is invalid.
    """
    x1, y1, x2, y2 = region

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid region: {region}. x2 must be > x1 and y2 must be > y1.")

    return image.crop((x1, y1, x2, y2))


def extract_all_documents(
    image: Image.Image,
    min_size: tuple[int, int] = (50, 50),
) -> list[Image.Image]:
    """
    Extract all document regions from an image.

    Args:
        image: PIL Image containing one or more documents.
        min_size: Minimum (width, height) for a valid document.

    Returns:
        List of cropped document images.
    """
    regions = detect_documents(image)
    documents = []

    for region in regions:
        x1, y1, x2, y2 = region
        width = x2 - x1
        height = y2 - y1

        # Filter out too-small regions
        if width >= min_size[0] and height >= min_size[1]:
            cropped = crop_document(image, region)
            documents.append(cropped)

    return documents


def process_multi_document(
    image: Image.Image,
    auto_orient: bool = True,
    min_size: tuple[int, int] = (50, 50),
) -> dict[str, Any]:
    """
    Process a multi-document image.

    Detects all documents, applies orientation correction to each,
    and returns processed images.

    Args:
        image: PIL Image to process.
        auto_orient: Whether to auto-correct orientation.
        min_size: Minimum document size.

    Returns:
        Dictionary with 'document_count' and 'documents' list.
    """
    documents = extract_all_documents(image, min_size=min_size)

    if auto_orient:
        processed_docs = []
        for doc in documents:
            corrected, _ = auto_correct_orientation(doc)
            processed_docs.append(corrected)
        documents = processed_docs

    return {
        "document_count": len(documents),
        "documents": documents,
    }


def preprocess_image(
    source: ImageInput,
    max_size: int = 1024,
    auto_orient: bool = True,
) -> Image.Image:
    """
    Full preprocessing pipeline for a single image.

    Args:
        source: Image source (path, PIL Image, or numpy array).
        max_size: Maximum dimension for resizing.
        auto_orient: Whether to auto-correct orientation.

    Returns:
        Preprocessed PIL Image.
    """
    # Load image
    image = load_image(source)

    # Normalize to RGB
    image = normalize_image(image)

    # Resize if needed
    image = resize_image(image, max_size=max_size)

    # Auto-correct orientation
    if auto_orient:
        image, _ = auto_correct_orientation(image)

    return image
