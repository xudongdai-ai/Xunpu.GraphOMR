from __future__ import annotations

import cv2
import numpy as np


def preprocess_gray(gray: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if gray.ndim != 2:
        raise ValueError("preprocess_gray expects a single-channel grayscale image")
    denoised = cv2.medianBlur(gray, 3)
    angle = estimate_skew_degrees(denoised)
    corrected = deskew(denoised, angle)
    return corrected, {"skew_degrees": float(angle)}


def estimate_skew_degrees(gray: np.ndarray) -> float:
    binary = binarize(gray)
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, gray.shape[1] // 8),
        minLineLength=max(30, gray.shape[1] // 5),
        maxLineGap=8,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in line]
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 12:
            angles.append(float(angle))
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(gray: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 0.25:
        return gray
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Return an inverted binary image: foreground ink is 255."""
    if gray.ndim != 2:
        raise ValueError("binarize expects a single-channel grayscale image")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def horizontal_lines(binary: np.ndarray, min_width: int | None = None) -> np.ndarray:
    height, width = binary.shape
    kernel_width = min_width or max(25, width // 18)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def vertical_lines(binary: np.ndarray, min_height: int) -> np.ndarray:
    kernel_height = max(5, int(min_height))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
