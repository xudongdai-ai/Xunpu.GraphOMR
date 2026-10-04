import cv2
import numpy as np
import pytest

from xunpu_graph_omr.vision.preprocess import estimate_skew_degrees


@pytest.mark.parametrize("shape", [(2, 1, 4), (2, 4)])
def test_hough_line_layouts(monkeypatch, shape):
    lines = np.array([[10, 10, 110, 12], [20, 20, 120, 22]], dtype=np.int32)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *a, **kw: lines.reshape(shape))
    assert estimate_skew_degrees(np.full((150, 150), 255, np.uint8)) == pytest.approx(
        np.degrees(np.arctan2(2, 100))
    )
