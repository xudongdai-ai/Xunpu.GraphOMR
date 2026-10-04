"""
OMR 前处理增强：把扫描/低分辨率位图清理成更适合 Audiveris 识别的图。
高 DPI 重渲染 → 去噪 → 纠偏 → CLAHE 对比度增强 → 适度锐化 → 输出 RGB PNG。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from xunpu_graph_omr.vision.preprocess import deskew, estimate_skew_degrees


def _load_gray(path: str | Path, render_dpi: int) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import fitz

        with fitz.open(path) as doc:
            page = doc[0]
            pix = page.get_pixmap(
                matrix=fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0),
                colorspace=fitz.csGRAY,
                alpha=False,
            )
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            return arr.copy()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def enhance_for_omr(
    input_path: str | Path,
    output_path: str | Path,
    render_dpi: int = 300,
    min_long_side: int = 2200,
) -> dict:
    """增强一张乐谱图并写出 PNG，返回处理参数。"""
    gray = _load_gray(input_path, render_dpi)

    h, w = gray.shape
    long_side = max(h, w)
    scaled = False
    if Path(input_path).suffix.lower() != ".pdf" and long_side < min_long_side:
        scale = min_long_side / long_side
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        scaled = True

    den = cv2.medianBlur(gray, 3)
    angle = estimate_skew_degrees(den)
    den = deskew(den, angle)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(den)
    blur = cv2.GaussianBlur(enh, (0, 0), 1.0)
    sharp = cv2.addWeighted(enh, 1.5, blur, -0.5, 0)

    # 输出 RGB(而非灰度),兼容 Audiveris 的图像输入
    cv2.imwrite(str(output_path), cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
    return {
        "skew_degrees": round(float(angle), 3),
        "out_size": (sharp.shape[1], sharp.shape[0]),
        "render_dpi": render_dpi,
        "scaled": scaled,
    }
