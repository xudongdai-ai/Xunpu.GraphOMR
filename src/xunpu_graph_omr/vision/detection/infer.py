"""
Faster R-CNN inference adapter.

Exposes the same surface as xunpu_graph_omr.vision.detectors.ClassicalSymbolDetector so it can
replace it in the pipeline once a checkpoint exists::

    from xunpu_graph_omr.vision.detection.infer import FasterRCNNSymbolDetector
    detector = FasterRCNNSymbolDetector("out/detector/best_detector.pt")
    symbols = detector.detect(gray, staves, notes, stems, measures)

Note: the detector runs on the *grayscale page image*, not the binarized mask —
pass `page.gray` (the pipeline currently passes `binary`; switch that line when
wiring this in). It also derives staff_index / measure_index geometrically, so
the `notes` / `stems` / `measures` arguments are accepted but unused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from xunpu_graph_omr.vision.detection.labels import ID_TO_TYPE
from xunpu_graph_omr.vision.detection.model import load_detector
from xunpu_graph_omr.model import MeasureInfo, NoteEvent, Staff, StemEvent, SymbolDetection


class FasterRCNNSymbolDetector:
    name = "fasterrcnn"

    def __init__(
        self,
        checkpoint: str | Path,
        score_thresh: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_detector(checkpoint, self.device)
        self.score_thresh = score_thresh

    @torch.no_grad()
    def detect(
        self,
        image,
        staves: tuple[Staff, ...],
        notes: tuple[NoteEvent, ...] = (),
        stems: tuple[StemEvent, ...] = (),
        measures: tuple[MeasureInfo, ...] = (),
    ) -> tuple[SymbolDetection, ...]:
        tensor = self._to_tensor(image).to(self.device)
        output = self.model([tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        detections: list[SymbolDetection] = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores, strict=False)):
            if score < self.score_thresh:
                continue
            sym_type = ID_TO_TYPE.get(int(label))
            if sym_type is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            bbox = (x0, y0, x1 - x0, y1 - y0)
            staff_idx = _assign_staff(y0, y1, staves)
            measure_idx = _assign_measure((x0 + x1) / 2.0, staff_idx, measures)
            detections.append(
                SymbolDetection(
                    id=f"{sym_type}:{i}",
                    type=sym_type,
                    bbox=bbox,
                    confidence=float(score),
                    staff_index=staff_idx,
                    measure_index=measure_idx,
                    source_ref=f"frcnn:{i}",
                )
            )
        return tuple(detections)

    @staticmethod
    def _to_tensor(image) -> torch.Tensor:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
        return t.permute(2, 0, 1).contiguous()


def _assign_staff(y0: float, y1: float, staves: tuple[Staff, ...]) -> int | None:
    if not staves:
        return None
    cy = (y0 + y1) / 2.0
    best, best_d = None, float("inf")
    for staff in staves:
        margin = staff.interline * 4
        if staff.top - margin <= cy <= staff.bottom + margin:
            d = abs(cy - (staff.top + staff.bottom) / 2.0)
            if d < best_d:
                best, best_d = staff.index, d
    return best


def _assign_measure(cx: float, staff_idx: int | None, measures: tuple[MeasureInfo, ...]) -> int | None:
    for m in measures:
        if m.staff_index == staff_idx and m.x_start <= cx < m.x_end:
            return m.index
    return None
