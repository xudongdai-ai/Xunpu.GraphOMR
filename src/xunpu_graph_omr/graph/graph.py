from __future__ import annotations

import math

from xunpu_graph_omr.model import BBox, GraphEdge, GraphNode, HeteroGraph, Staff, SymbolDetection


NODE_TYPES = (
    "notehead_filled",
    "notehead_empty",
    "stem",
    "beam",
    "flag",
    "rest",
    "dot",
    "accidental",
    "clef",
    "key_signature",
    "time_signature",
    "barline",
    "staff",
    "measure",
)

EDGE_TYPES = (
    "belongs_to_staff",
    "notehead_to_stem",
    "stem_to_beam",
    "notehead_to_accidental",
    "notehead_to_dot",
    "same_chord",
    "same_voice_next",
    "inside_measure",
    "clef_affects_staff",
    "key_affects_measure",
    "tie_connects",
    "slur_connects",
)


def build_heterograph(
    symbols: tuple[SymbolDetection, ...],
    staves: tuple[Staff, ...],
    width: int,
    height: int,
) -> HeteroGraph:
    staff_by_index = {staff.index: staff for staff in staves}
    nodes = tuple(_node_from_symbol(symbol, staff_by_index, width, height) for symbol in symbols)
    node_by_id = {node.id: node for node in nodes}
    edges: list[GraphEdge] = []

    staff_nodes = {node.staff_index: node for node in nodes if node.type == "staff" and node.staff_index is not None}
    measure_nodes = [node for node in nodes if node.type == "measure"]
    note_nodes = [node for node in nodes if node.type.startswith("notehead")]
    stem_nodes = [node for node in nodes if node.type == "stem"]

    for node in nodes:
        staff_node = staff_nodes.get(node.staff_index)
        if staff_node and node.id != staff_node.id:
            edges.append(_edge(node, staff_node, "belongs_to_staff"))

        if node.type != "measure":
            measure_node = _containing_measure(node, measure_nodes)
            if measure_node:
                edges.append(_edge(node, measure_node, "inside_measure"))

    for note in note_nodes:
        stem = _nearest_stem(note, stem_nodes, staff_by_index)
        if stem:
            edges.append(_edge(note, stem, "notehead_to_stem"))

    for left, right in zip(sorted(note_nodes, key=lambda n: (n.staff_index or 0, n.measure_index or 0, _cx(n.bbox))), sorted(note_nodes, key=lambda n: (n.staff_index or 0, n.measure_index or 0, _cx(n.bbox)))[1:], strict=False):
        if left.staff_index == right.staff_index and left.measure_index == right.measure_index:
            if abs(_cx(left.bbox) - _cx(right.bbox)) < _staff_interline(left.staff_index, staff_by_index) * 0.75:
                edges.append(_edge(left, right, "same_chord"))
            elif _cx(left.bbox) < _cx(right.bbox):
                edges.append(_edge(left, right, "same_voice_next"))

    return HeteroGraph(nodes=nodes, edges=tuple(_dedupe_edges(edges)))


def _node_from_symbol(
    symbol: SymbolDetection,
    staff_by_index: dict[int, Staff],
    width: int,
    height: int,
) -> GraphNode:
    staff = staff_by_index.get(symbol.staff_index) if symbol.staff_index is not None else None
    features = _node_features(symbol.bbox, symbol.confidence, width, height, staff)
    return GraphNode(
        id=symbol.id,
        type=symbol.type,
        bbox=symbol.bbox,
        confidence=symbol.confidence,
        staff_index=symbol.staff_index,
        measure_index=symbol.measure_index,
        features=features,
        source_ref=symbol.source_ref,
    )


def _node_features(bbox: BBox, confidence: float, width: int, height: int, staff: Staff | None) -> dict[str, float]:
    x, y, w, h = bbox
    y_mid = y + h / 2.0
    staff_position = 0.0
    if staff:
        staff_middle = staff.line_ys[2]
        staff_position = float(round(2.0 * (staff_middle - y_mid) / staff.interline))
    return {
        "x_norm": x / max(1.0, width),
        "y_norm": y / max(1.0, height),
        "w_norm": w / max(1.0, width),
        "h_norm": h / max(1.0, height),
        "area_norm": (w * h) / max(1.0, width * height),
        "aspect": w / max(1.0, h),
        "staff_position": staff_position,
        "confidence": confidence,
    }


def _edge(source: GraphNode, target: GraphNode, edge_type: str) -> GraphEdge:
    return GraphEdge(
        source=source.id,
        target=target.id,
        type=edge_type,
        confidence=min(source.confidence, target.confidence),
        features=_edge_features(source, target),
    )


def _edge_features(source: GraphNode, target: GraphNode) -> dict[str, float]:
    sx, sy = _center(source.bbox)
    tx, ty = _center(target.bbox)
    staff_scale = max(1.0, math.sqrt(source.bbox[2] * source.bbox[3]))
    dx = (tx - sx) / staff_scale
    dy = (ty - sy) / staff_scale
    return {
        "dx": dx,
        "dy": dy,
        "distance": math.hypot(dx, dy),
        "iou": _iou(source.bbox, target.bbox),
        "theta": math.atan2(ty - sy, tx - sx),
        "delta_staff_position": target.features.get("staff_position", 0.0) - source.features.get("staff_position", 0.0),
        "same_staff": 1.0 if source.staff_index == target.staff_index else 0.0,
    }


def _containing_measure(node: GraphNode, measures: list[GraphNode]) -> GraphNode | None:
    cx = _cx(node.bbox)
    candidates = [
        measure
        for measure in measures
        if measure.staff_index == node.staff_index and measure.bbox[0] <= cx < measure.bbox[0] + measure.bbox[2]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda measure: abs(_cx(measure.bbox) - cx))


def _nearest_stem(
    note: GraphNode,
    stems: list[GraphNode],
    staff_by_index: dict[int, Staff],
) -> GraphNode | None:
    interline = _staff_interline(note.staff_index, staff_by_index)
    candidates = [
        stem
        for stem in stems
        if stem.staff_index == note.staff_index
        and abs(_cx(stem.bbox) - _cx(note.bbox)) <= interline * 0.9
        and stem.bbox[1] <= _cy(note.bbox) + interline
        and stem.bbox[1] + stem.bbox[3] >= _cy(note.bbox) - interline
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda stem: abs(_cx(stem.bbox) - _cx(note.bbox)))


def _staff_interline(staff_index: int | None, staff_by_index: dict[int, Staff]) -> float:
    if staff_index is None or staff_index not in staff_by_index:
        return 12.0
    return staff_by_index[staff_index].interline


def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    seen: set[tuple[str, str, str]] = set()
    kept: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.type)
        if key in seen:
            continue
        seen.add(key)
        kept.append(edge)
    return kept


def _center(bbox: BBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)


def _cx(bbox: BBox) -> float:
    return bbox[0] + bbox[2] / 2.0


def _cy(bbox: BBox) -> float:
    return bbox[1] + bbox[3] / 2.0


def _iou(a: BBox, b: BBox) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0
