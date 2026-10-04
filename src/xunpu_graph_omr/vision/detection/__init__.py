"""Learned symbol detection (Faster R-CNN) for printed + handwritten scores."""

from xunpu_graph_omr.vision.detection.labels import (
    DETECTION_TYPES,
    ID_TO_TYPE,
    NUM_CLASSES,
    TYPE_TO_ID,
)

__all__ = ["DETECTION_TYPES", "ID_TO_TYPE", "NUM_CLASSES", "TYPE_TO_ID"]
