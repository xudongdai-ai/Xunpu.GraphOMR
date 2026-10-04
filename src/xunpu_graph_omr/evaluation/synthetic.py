from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from xunpu_graph_omr.vision.image_io import ensure_parent, write_json
from xunpu_graph_omr.model import Staff
from xunpu_graph_omr.vision.pitch import pitch_to_y

DEFAULT_NOTES = (
    ("F", 4),
    ("A", 4),
    ("C", 5),
    ("E", 5),
    ("G", 4),
    ("B", 4),
    ("D", 5),
    ("F", 5),
)

DurationSpec = tuple[str, int, str]


def draw_simple_score(
    image_path: str | Path,
    truth_path: str | Path | None = None,
    notes: tuple[tuple[str, int], ...] = DEFAULT_NOTES,
) -> dict:
    width = 980
    height = 260
    x_start = 80
    x_end = 900
    top = 80
    interline = 16
    staff = Staff(
        index=0,
        line_ys=tuple(float(top + i * interline) for i in range(5)),  # type: ignore[arg-type]
        x_start=x_start,
        x_end=x_end,
    )

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for y in staff.line_ys:
        draw.line((x_start, y, x_end, y), fill=0, width=2)

    x_positions = [140 + i * 88 for i in range(len(notes))]
    bar_x = 140 + 4 * 88 - 38
    draw.line((bar_x, staff.top, bar_x, staff.bottom), fill=0, width=3)
    draw.line((x_end, staff.top, x_end, staff.bottom), fill=0, width=3)

    truth_notes = []
    for index, ((step, octave), x) in enumerate(zip(notes, x_positions, strict=True), start=1):
        y = pitch_to_y(step, octave, staff)
        rx = interline * 0.58
        ry = interline * 0.42
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=0)
        stem_up = y >= staff.line_ys[2]
        if stem_up:
            stem_x = x + rx - 1
            draw.line((stem_x, y, stem_x, y - interline * 3.1), fill=0, width=2)
        else:
            stem_x = x - rx + 1
            draw.line((stem_x, y, stem_x, y + interline * 3.1), fill=0, width=2)
        truth_notes.append(
            {
                "index": index,
                "x": x,
                "step": step,
                "octave": octave,
                "pitch_name": f"{step}{octave}",
                "measure": 1 if x < bar_x else 2,
            }
        )

    ensure_parent(image_path)
    image.save(image_path)

    truth = {
        "image": str(image_path),
        "clef": "treble",
        "notes": truth_notes,
    }
    if truth_path:
        write_json(truth_path, truth)
    return truth


def draw_duration_score(
    image_path: str | Path,
    truth_path: str | Path | None = None,
) -> dict:
    width = 760
    height = 240
    x_start = 80
    x_end = 680
    top = 76
    interline = 16
    staff = Staff(
        index=0,
        line_ys=tuple(float(top + i * interline) for i in range(5)),  # type: ignore[arg-type]
        x_start=x_start,
        x_end=x_end,
    )
    notes: tuple[DurationSpec, ...] = (
        ("F", 4, "whole"),
        ("A", 4, "half"),
        ("C", 5, "half"),
    )
    durations = {"whole": 4, "half": 2, "quarter": 1}
    x_positions = (210, 450, 570)
    bar_x = 350

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for y in staff.line_ys:
        draw.line((x_start, y, x_end, y), fill=0, width=2)
    draw.line((bar_x, staff.top, bar_x, staff.bottom), fill=0, width=3)
    draw.line((x_end, staff.top, x_end, staff.bottom), fill=0, width=3)

    truth_notes = []
    for index, ((step, octave, note_type), x) in enumerate(zip(notes, x_positions, strict=True), start=1):
        y = pitch_to_y(step, octave, staff)
        rx = interline * 0.62
        ry = interline * 0.44
        if note_type == "quarter":
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=0)
        else:
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=0, width=2)
        if note_type != "whole":
            stem_up = y >= staff.line_ys[2]
            if stem_up:
                stem_x = x + rx - 1
                draw.line((stem_x, y, stem_x, y - interline * 3.1), fill=0, width=2)
            else:
                stem_x = x - rx + 1
                draw.line((stem_x, y, stem_x, y + interline * 3.1), fill=0, width=2)
        truth_notes.append(
            {
                "index": index,
                "x": x,
                "step": step,
                "octave": octave,
                "pitch_name": f"{step}{octave}",
                "note_type": note_type,
                "duration": durations[note_type],
                "measure": 1 if x < bar_x else 2,
            }
        )

    ensure_parent(image_path)
    image.save(image_path)
    truth = {"image": str(image_path), "clef": "treble", "notes": truth_notes}
    if truth_path:
        write_json(truth_path, truth)
    return truth
