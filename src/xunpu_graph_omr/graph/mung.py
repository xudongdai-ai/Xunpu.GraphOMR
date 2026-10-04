"""
Pure-stdlib MUSCIMA++ MUNG parsing + class→type mapping.

Kept dependency-free (only xml/dataclasses) so both the GNN path
(xunpu_graph_omr.graph.muscima_dataset, needs torch_geometric) and the detector path
(xunpu_graph_omr.vision.detection, needs torchvision) can reuse it without dragging in the
other's heavy dependency.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# MUSCIMA++ class-name → coarse node/symbol type
MUSCIMA_TO_TYPE: dict[str, str] = {
    # noteheads
    "noteheadFull": "notehead_filled",
    "noteheadFullSmall": "notehead_filled",
    "noteheadHalf": "notehead_empty",
    "noteheadWhole": "notehead_empty",
    "noteheadWholeLarge": "notehead_empty",
    # stems
    "stem": "stem",
    # beams & flags
    "beam": "beam",
    "flag8thUp": "flag",
    "flag8thDown": "flag",
    "flag16thUp": "flag",
    "flag16thDown": "flag",
    "flag32ndUp": "flag",
    "flag32ndDown": "flag",
    "flag64thUp": "flag",
    "flag64thDown": "flag",
    # accidentals
    "sharp": "accidental",
    "flat": "accidental",
    "natural": "accidental",
    "doubleSharp": "accidental",
    "doubleFlat": "accidental",
    # augmentation dots
    "augmentationDot": "dot",
    # clefs
    "trebleClef": "clef",
    "bassClef": "clef",
    "altoClef": "clef",
    "tenorClef": "clef",
    "sopranoClef": "clef",
    # rests
    "restWhole": "rest",
    "restHalf": "rest",
    "restQuarter": "rest",
    "rest8th": "rest",
    "rest16th": "rest",
    "rest32nd": "rest",
    "rest64th": "rest",
    "multiMeasureRest": "rest",
    # barlines
    "barline": "barline",
    "measureSeparator": "barline",
    "repeatMeasure": "barline",
    # key / time signatures
    "keySignature": "key_signature",
    "timeSignature": "time_signature",
}

NODE_TYPES: list[str] = sorted(set(MUSCIMA_TO_TYPE.values()))


@dataclass
class MungNode:
    node_id: int
    class_name: str
    top: float
    left: float
    width: float
    height: float
    outlinks: list[int] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return self.left + self.width / 2

    @property
    def cy(self) -> float:
        return self.top + self.height / 2


def parse_mung_xml(xml_path: str | Path) -> list[MungNode]:
    """Return all Nodes from one MUSCIMA++ MUNG annotation XML."""
    root = ET.parse(xml_path).getroot()
    nodes: list[MungNode] = []
    for elem in root.iter("Node"):
        outlinks_text = (elem.findtext("Outlinks") or "").strip()
        nodes.append(
            MungNode(
                node_id=int(elem.findtext("Id", "0")),
                class_name=(elem.findtext("ClassName") or "").strip(),
                top=float(elem.findtext("Top", "0")),
                left=float(elem.findtext("Left", "0")),
                width=float(elem.findtext("Width", "1")),
                height=float(elem.findtext("Height", "1")),
                outlinks=[int(x) for x in outlinks_text.split() if x],
            )
        )
    return nodes
