"""
Unified detection label space for printed (DoReMi) and handwritten (MUSCIMA++) data.

Both datasets are mapped onto the SAME 12 coarse symbol types so one Faster R-CNN
can be trained on either or both.  The type names match xunpu_graph_omr.graph.graph.NODE_TYPES
(minus the structural ``staff`` / ``measure`` which are derived geometrically, not
detected), so detector output can flow straight into build_heterograph().

Label ids: 0 = __background__, 1..12 = the types below (torchvision convention).
"""
from __future__ import annotations

from xunpu_graph_omr.graph.mung import MUSCIMA_TO_TYPE

# 12 detectable glyph types (sorted for a stable id assignment)
DETECTION_TYPES: list[str] = [
    "accidental",
    "barline",
    "beam",
    "clef",
    "dot",
    "flag",
    "key_signature",
    "notehead_empty",
    "notehead_filled",
    "rest",
    "stem",
    "time_signature",
]

# torchvision wants 0 reserved for background
TYPE_TO_ID: dict[str, int] = {t: i + 1 for i, t in enumerate(DETECTION_TYPES)}
ID_TO_TYPE: dict[int, str] = {i: t for t, i in TYPE_TO_ID.items()}
NUM_CLASSES: int = len(DETECTION_TYPES) + 1  # + background


def muscima_class_to_type(class_name: str) -> str | None:
    """MUSCIMA++ MUNG ClassName → coarse type (reuses the GNN mapping)."""
    return MUSCIMA_TO_TYPE.get(class_name)


def doremi_class_to_type(class_name: str) -> str | None:
    """
    DoReMi SMuFL-style category name → coarse type.

    Uses keyword rules instead of an exhaustive table so it tolerates the long
    tail of DoReMi class names (flag8thUp, accidentalDoubleFlat, restQuarter…).
    Returns None for classes we don't model (slurs, dynamics, text, ties…).
    """
    name = class_name.strip()
    low = name.lower()

    # order matters: more specific checks first
    if "notehead" in low:
        if any(k in low for k in ("black", "full")):
            return "notehead_filled"
        if any(k in low for k in ("half", "whole", "double")):
            return "notehead_empty"
        return "notehead_filled"
    if low == "stem" or low.endswith("stem"):
        return "stem"
    if "beam" in low:
        return "beam"
    if "flag" in low:
        return "flag"
    if low.startswith("accidental") or low in {"sharp", "flat", "natural", "doublesharp", "doubleflat"}:
        return "accidental"
    if "augmentationdot" in low or low == "dot":
        return "dot"
    if "clef" in low:
        return "clef"
    if low.startswith("rest") or "rest" in low:
        return "rest"
    if "keysig" in low or "keysignature" in low:
        return "key_signature"
    if "timesig" in low or "timesignature" in low:
        return "time_signature"
    if "barline" in low:
        return "barline"
    return None
