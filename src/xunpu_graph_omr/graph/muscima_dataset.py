"""
Parses MUSCIMA++ v2 annotation XMLs (MUNG format) into PyTorch Geometric HeteroData.

Expected directory layout
--------------------------
muscima_pp/
  v2.0/
    data/
      annotations/    *.xml   (MUNG format, one file per score page)
      images/          *.png   (optional; size is inferred from bboxes if absent)

Run::

    from xunpu_graph_omr.graph.muscima_dataset import MuscimaPlusPlusDataset
    ds = MuscimaPlusPlusDataset("/path/to/muscima_pp/v2.0/data", split="train")
    data = ds[0]   # torch_geometric.data.HeteroData

Optional dependency — install with:
    pip install "xunpu-graph-omr[gnn]"
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from xunpu_graph_omr.graph.mung import MUSCIMA_TO_TYPE, NODE_TYPES, MungNode, parse_mung_xml

# --------------------------------------------------------------------------- #
# 1.  MUSCIMA++ class-name → coarse node-type mapping lives in xunpu_graph_omr.graph.mung   #
#     (MUSCIMA_TO_TYPE, NODE_TYPES, MungNode, parse_mung_xml imported above)  #
# --------------------------------------------------------------------------- #

# Which (src_type, dst_type) pairs are semantically plausible and get candidate edges
CANDIDATE_PAIRS: list[tuple[str, str]] = [
    ("notehead_filled", "stem"),
    ("notehead_empty", "stem"),
    ("notehead_filled", "accidental"),
    ("notehead_empty", "accidental"),
    ("notehead_filled", "dot"),
    ("notehead_empty", "dot"),
    ("stem", "beam"),
    ("stem", "flag"),
    ("clef", "key_signature"),
    ("key_signature", "time_signature"),
]

# KNN k and max distance (as fraction of image diagonal) for candidate generation
_KNN_K = 5
_MAX_DIST_FRAC = 0.12

NODE_FEAT_DIM = 8
EDGE_FEAT_DIM = 5


# --------------------------------------------------------------------------- #
# 2.  Feature extractors                                                       #
# --------------------------------------------------------------------------- #

def _node_feat(node: MungNode, img_w: float, img_h: float) -> list[float]:
    return [
        node.left / img_w,
        node.top / img_h,
        node.width / img_w,
        node.height / img_h,
        (node.width * node.height) / (img_w * img_h),
        node.width / max(1.0, node.height),   # aspect ratio
        node.cx / img_w,
        node.cy / img_h,
    ]


def _edge_feat(src: MungNode, dst: MungNode, img_w: float, img_h: float) -> list[float]:
    diag = math.hypot(img_w, img_h) or 1.0
    dx = (dst.cx - src.cx) / img_w
    dy = (dst.cy - src.cy) / img_h
    dist = math.hypot(dst.cx - src.cx, dst.cy - src.cy) / diag
    # IoU
    ax0, ay0, ax1, ay1 = src.left, src.top, src.left + src.width, src.top + src.height
    bx0, by0, bx1, by1 = dst.left, dst.top, dst.left + dst.width, dst.top + dst.height
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = src.width * src.height + dst.width * dst.height - inter
    iou = inter / union if union > 0 else 0.0
    return [dx, dy, dist, iou, math.atan2(dy, dx) / math.pi]


# --------------------------------------------------------------------------- #
# 4.  Convert one parsed page to HeteroData                                   #
# --------------------------------------------------------------------------- #

def mung_to_heterodata(mung_nodes: list[MungNode], img_w: float, img_h: float) -> HeteroData:
    """
    Build a PyG HeteroData from a list of MungNodes.

    For each CANDIDATE_PAIR (src_type, dst_type), candidate edges are the K
    nearest dst nodes within the distance threshold.  Ground-truth edges from
    MUNG Outlinks are always included.  Labels: 1 = edge exists, 0 = does not.
    """
    known = [n for n in mung_nodes if n.class_name in MUSCIMA_TO_TYPE]

    # group by coarse type, build per-type local index
    type_to_nodes: dict[str, list[MungNode]] = {t: [] for t in NODE_TYPES}
    node_to_local: dict[int, tuple[str, int]] = {}   # node_id → (type, local_idx)
    for node in known:
        ntype = MUSCIMA_TO_TYPE[node.class_name]
        local_idx = len(type_to_nodes[ntype])
        type_to_nodes[ntype].append(node)
        node_to_local[node.node_id] = (ntype, local_idx)

    # undirected ground-truth edge set (node_id pairs, both directions)
    gt_pairs: set[tuple[int, int]] = set()
    for node in known:
        for dst_id in node.outlinks:
            if dst_id in node_to_local:
                gt_pairs.add((node.node_id, dst_id))
                gt_pairs.add((dst_id, node.node_id))

    diag = math.hypot(img_w, img_h) or 1.0
    max_px = _MAX_DIST_FRAC * diag

    data = HeteroData()

    # node features
    for ntype, nodes in type_to_nodes.items():
        if nodes:
            data[ntype].x = torch.tensor(
                [_node_feat(n, img_w, img_h) for n in nodes], dtype=torch.float
            )
        else:
            data[ntype].x = torch.zeros((0, NODE_FEAT_DIM), dtype=torch.float)

    # candidate edges
    for src_type, dst_type in CANDIDATE_PAIRS:
        src_nodes = type_to_nodes[src_type]
        dst_nodes = type_to_nodes[dst_type]
        rel = (src_type, "to", dst_type)

        if not src_nodes or not dst_nodes:
            data[rel].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[rel].edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)
            data[rel].y = torch.zeros(0, dtype=torch.long)
            continue

        src_idxs, dst_idxs, attrs, labels = [], [], [], []

        for si, src in enumerate(src_nodes):
            # KNN within distance threshold
            by_dist = sorted(
                enumerate(dst_nodes),
                key=lambda t: math.hypot(t[1].cx - src.cx, t[1].cy - src.cy),
            )
            knn_dis = {di for di, dst in by_dist[:_KNN_K]
                       if math.hypot(dst.cx - src.cx, dst.cy - src.cy) <= max_px}

            # always include GT edges regardless of distance
            gt_dis = {
                di for di, dst in enumerate(dst_nodes)
                if (src.node_id, dst.node_id) in gt_pairs
            }

            for di in knn_dis | gt_dis:
                dst = dst_nodes[di]
                src_idxs.append(si)
                dst_idxs.append(di)
                attrs.append(_edge_feat(src, dst, img_w, img_h))
                labels.append(int((src.node_id, dst.node_id) in gt_pairs))

        data[rel].edge_index = torch.tensor([src_idxs, dst_idxs], dtype=torch.long)
        data[rel].edge_attr = torch.tensor(attrs, dtype=torch.float).reshape(-1, EDGE_FEAT_DIM)
        data[rel].y = torch.tensor(labels, dtype=torch.long)

    return data


# --------------------------------------------------------------------------- #
# 5.  InMemoryDataset wrapper                                                  #
# --------------------------------------------------------------------------- #

class MuscimaPlusPlusDataset(InMemoryDataset):
    """
    Wraps a MUSCIMA++ annotations directory as a PyG InMemoryDataset.

    Parameters
    ----------
    root : str | Path
        Directory that contains ``annotations/*.xml`` (and optionally ``images/*.png``).
    split : "train" | "val" | "test"
        80 / 10 / 10 split by sorted filename order.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None) -> None:
        self.split = split
        super().__init__(str(root), transform=transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / "annotations")

    @property
    def raw_file_names(self) -> list[str]:
        return [p.name for p in sorted(Path(self.raw_dir).glob("*.xml"))]

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed" / self.split)

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def process(self) -> None:
        xml_paths = sorted(Path(self.raw_dir).glob("*.xml"))
        n = len(xml_paths)
        if self.split == "train":
            xml_paths = xml_paths[: int(0.8 * n)]
        elif self.split == "val":
            xml_paths = xml_paths[int(0.8 * n) : int(0.9 * n)]
        else:
            xml_paths = xml_paths[int(0.9 * n) :]

        data_list = []
        for xml_path in xml_paths:
            mung_nodes = parse_mung_xml(xml_path)
            img_w, img_h = _infer_image_size(mung_nodes)
            data_list.append(mung_to_heterodata(mung_nodes, img_w, img_h))

        self.save(data_list, self.processed_paths[0])


def _infer_image_size(nodes: list[MungNode]) -> tuple[float, float]:
    if not nodes:
        return 800.0, 1000.0
    margin = 50.0
    return (
        max(n.left + n.width for n in nodes) + margin,
        max(n.top + n.height for n in nodes) + margin,
    )
