"""
Detection datasets for DoReMi (printed, COCO) and MUSCIMA++ (handwritten, MUNG).

Both yield the torchvision detection format::

    image:  FloatTensor [3, H, W] in 0..1
    target: {"boxes": FloatTensor[N,4] (x0,y0,x1,y1),
             "labels": Int64Tensor[N],
             "image_id": Int64Tensor[1]}

Boxes use absolute pixel xyxy.  Labels use detection.labels.TYPE_TO_ID.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from xunpu_graph_omr.vision.detection.labels import (
    TYPE_TO_ID,
    doremi_class_to_type,
    muscima_class_to_type,
)
from xunpu_graph_omr.graph.mung import parse_mung_xml


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.frombuffer(img.tobytes(), dtype=torch.uint8).float() / 255.0
    return arr.view(img.height, img.width, 3).permute(2, 0, 1).contiguous()


def _empty_target(image_id: int) -> dict:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float),
        "labels": torch.zeros((0,), dtype=torch.long),
        "image_id": torch.tensor([image_id]),
    }


# --------------------------------------------------------------------------- #
# DoReMi (printed) — COCO-format annotations                                  #
# --------------------------------------------------------------------------- #

class DoReMiDataset(Dataset):
    """
    Parameters
    ----------
    images_dir : directory of page images.
    coco_json  : COCO annotation file (instances) with categories+annotations.
    """

    def __init__(self, images_dir: str | Path, coco_json: str | Path) -> None:
        self.images_dir = Path(images_dir)
        coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))

        self.cat_id_to_type = {}
        for cat in coco["categories"]:
            t = doremi_class_to_type(cat["name"])
            if t is not None:
                self.cat_id_to_type[cat["id"]] = t

        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # keep only images that actually have at least one mappable box
        self.ids = [
            img_id for img_id in self.images
            if any(a["category_id"] in self.cat_id_to_type
                   for a in self.anns_by_image.get(img_id, []))
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.images[img_id]
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")

        boxes, labels = [], []
        for ann in self.anns_by_image.get(img_id, []):
            t = self.cat_id_to_type.get(ann["category_id"])
            if t is None:
                continue
            x, y, w, h = ann["bbox"]            # COCO xywh
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(TYPE_TO_ID[t])

        target = _empty_target(img_id)
        if boxes:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float)
            target["labels"] = torch.tensor(labels, dtype=torch.long)
        return _to_tensor(img), target


# --------------------------------------------------------------------------- #
# MUSCIMA++ (handwritten) — MUNG XML annotations                              #
# --------------------------------------------------------------------------- #

class MuscimaDetectionDataset(Dataset):
    """
    Parameters
    ----------
    annotations_dir : directory of MUNG ``*.xml`` files.
    images_dir      : directory of page images; each annotation ``foo.xml`` is
                      matched to ``images_dir/foo.png`` by default.
    image_suffix    : extension of the page images.
    """

    def __init__(
        self,
        annotations_dir: str | Path,
        images_dir: str | Path,
        image_suffix: str = ".png",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.suffix = image_suffix
        ann_dir = Path(annotations_dir)
        self.items: list[Path] = []
        for xml in sorted(ann_dir.glob("*.xml")):
            if (self.images_dir / f"{xml.stem}{image_suffix}").exists():
                self.items.append(xml)
        if not self.items:
            raise FileNotFoundError(
                f"No MUNG xml matched an image in {images_dir} "
                f"(expected {images_dir}/<xml-stem>{image_suffix})"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        xml = self.items[idx]
        img = Image.open(self.images_dir / f"{xml.stem}{self.suffix}").convert("RGB")

        boxes, labels = [], []
        for node in parse_mung_xml(xml):
            t = muscima_class_to_type(node.class_name)
            if t is None or node.width <= 0 or node.height <= 0:
                continue
            boxes.append([node.left, node.top, node.left + node.width, node.top + node.height])
            labels.append(TYPE_TO_ID[t])

        target = _empty_target(idx)
        if boxes:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float)
            target["labels"] = torch.tensor(labels, dtype=torch.long)
        return _to_tensor(img), target


def collate_fn(batch):
    """torchvision detection models take lists, not stacked tensors."""
    return tuple(zip(*batch, strict=False))
