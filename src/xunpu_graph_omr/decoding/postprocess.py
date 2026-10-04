"""
识别后处理:剔除音域离群的假音符(专治歌词/噪点被误识别成音符)。
用 IQR 离群检测,带保护阈值(误删比例过高则放弃,不乱删真音符)。
"""
from __future__ import annotations

from xunpu_graph_omr.export.musicxml_parse import ParsedNote

_STEP_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _midi(n: ParsedNote) -> int:
    return (n.octave + 1) * 12 + _STEP_SEMITONE.get(n.step.upper(), 0) + n.alter


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def filter_outlier_notes(
    notes: list[ParsedNote],
    iqr_k: float = 1.5,
    min_margin_semitones: float = 6.0,
    max_remove_frac: float = 0.35,
    min_notes: int = 8,
) -> tuple[list[ParsedNote], dict]:
    """返回 (过滤后的音符, 报告)。"""
    pitched = [n for n in notes if not n.is_rest]
    if len(pitched) < min_notes:
        return notes, {"removed": 0, "reason": "too_few_notes"}

    midis = sorted(_midi(n) for n in pitched)
    q1 = _percentile(midis, 0.25)
    q3 = _percentile(midis, 0.75)
    iqr = q3 - q1
    margin = max(iqr_k * iqr, min_margin_semitones)
    lo, hi = q1 - margin, q3 + margin

    kept = [n for n in notes if n.is_rest or (lo <= _midi(n) <= hi)]
    removed = len(notes) - len(kept)

    if removed > len(pitched) * max_remove_frac:
        return notes, {"removed": 0, "reason": "too_many_flagged_skipped",
                       "would_remove": removed, "range": (lo, hi)}

    return kept, {"removed": removed, "kept": len(kept), "range_midi": (lo, hi)}
