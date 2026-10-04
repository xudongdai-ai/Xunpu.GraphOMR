"""Faster R-CNN factory for OMR symbol detection."""
from __future__ import annotations

from pathlib import Path

import torch

from xunpu_graph_omr.vision.detection.labels import NUM_CLASSES


def build_fasterrcnn(num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True):
    """
    ResNet-50 FPN Faster R-CNN with the box head resized to ``num_classes``.

    Music symbols are small and dense, so the default anchor sizes are biased
    toward smaller objects.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

    weights = "DEFAULT" if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=weights)

    # smaller anchors for dense small glyphs
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.25, 0.5, 1.0, 2.0),) * len(anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model.rpn.head = RPNHead(
        model.backbone.out_channels,
        model.rpn.anchor_generator.num_anchors_per_location()[0],
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_detector(checkpoint: str | Path, device: torch.device | None = None):
    """Load a trained Faster R-CNN from a checkpoint saved by train_detector.py."""
    device = device or torch.device("cpu")
    model = build_fasterrcnn(pretrained_backbone=False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
