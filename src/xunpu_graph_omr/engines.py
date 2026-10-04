"""
Unified recognition-engine interface.

The GUI talks only to this layer, so the product (xunpu-graph-omr) can switch its
recognition backend without caring how each engine works:

    xunpu     — the in-repo pipeline (rule-based; experimental baseline)
    oemer     — optional, separately installed deep-learning baseline
    audiveris — optional local Audiveris baseline (AGPL, local only)

Every engine returns an EngineResult that always carries a MusicXML path on disk
and a flat list[ParsedNote] for display/export. Only the `xunpu` engine also
returns a full RecognitionResult (overlay, staves, measures, step log); external
engines give MusicXML only, so the GUI shows a reduced view for them.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from xunpu_graph_omr.audiveris import find_audiveris, run_audiveris_export
from xunpu_graph_omr.model import RecognitionResult
from xunpu_graph_omr.export.musicxml import write_musicxml
from xunpu_graph_omr.export.musicxml_parse import ParsedNote, measure_count, parse_musicxml
from xunpu_graph_omr.pipeline import recognize_path


@dataclass
class EngineResult:
    engine: str
    image_path: str
    musicxml_path: Path
    notes: list[ParsedNote]
    recognition: RecognitionResult | None = None
    log: str = ""

    @property
    def measure_count(self) -> int:
        if self.recognition is not None:
            return len(self.recognition.measures)
        return measure_count(self.notes)


class Engine:
    key = "base"
    label = "Base"

    def available(self) -> bool:
        raise NotImplementedError

    def unavailable_hint(self) -> str:
        return ""

    def recognize(self, image_path: str | Path, work_dir: str | Path) -> EngineResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Xunpu pipeline                                                                 #
# --------------------------------------------------------------------------- #

class XunpuEngine(Engine):
    key = "xunpu"
    label = "自研管线 (实验)"

    def available(self) -> bool:
        return True

    def recognize(self, image_path: str | Path, work_dir: str | Path) -> EngineResult:
        image_path = Path(image_path)
        result = recognize_path(image_path)
        xml = Path(work_dir) / f"{image_path.stem}.xunpu.musicxml"
        write_musicxml(result, xml, title=image_path.stem)
        notes = [
            ParsedNote(
                step=n.step, octave=n.octave, alter=0, measure=n.measure,
                is_rest=False, note_type=n.note_type,
            )
            for n in sorted(result.notes, key=lambda n: (n.measure, n.staff_index, n.x))
        ]
        return EngineResult(self.key, str(image_path), xml, notes, recognition=result)


# --------------------------------------------------------------------------- #
# external engines (subprocess → MusicXML)                                     #
# --------------------------------------------------------------------------- #

def _newest_musicxml(folder: Path) -> Path | None:
    candidates = sorted(
        list(folder.rglob("*.musicxml")) + list(folder.rglob("*.mxl")) + list(folder.rglob("*.xml")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return candidates[0] if candidates else None


class OemerEngine(Engine):
    key = "oemer"
    label = "oemer (外部深度学习引擎)"

    def available(self) -> bool:
        return shutil.which("oemer") is not None

    def unavailable_hint(self) -> str:
        return "未检测到 oemer，请先安装：pip install oemer（首次运行会自动下载模型）"

    def recognize(self, image_path: str | Path, work_dir: str | Path) -> EngineResult:
        image_path = Path(image_path)
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["oemer", "-o", str(work), str(image_path)],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=600,
        )
        xml = _newest_musicxml(work)
        if completed.returncode != 0 or xml is None:
            raise RuntimeError(f"oemer 失败 (code {completed.returncode}):\n{completed.stdout[-3000:]}")
        return EngineResult(self.key, str(image_path), xml, parse_musicxml(xml), log=completed.stdout[-3000:])


class AudiverisEngine(Engine):
    # Optional, separately installed baseline; never presented as our own model.
    key = "audiveris"
    label = "Audiveris (外部基线)"

    def available(self) -> bool:
        return find_audiveris() is not None

    def unavailable_hint(self) -> str:
        return "未找到内置识别引擎；如为源码运行，请把 Audiveris.app 放到项目 vendor/ 目录，或安装到 /Applications。"

    def recognize(self, image_path: str | Path, work_dir: str | Path) -> EngineResult:
        image_path = Path(image_path)
        xml = run_audiveris_export(image_path, work_dir)
        return EngineResult(self.key, str(image_path), xml, parse_musicxml(xml))


# --------------------------------------------------------------------------- #
# registry                                                                     #
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, Engine] = {e.key: e for e in (AudiverisEngine(), OemerEngine(), XunpuEngine())}

# The independent pipeline is the default; baselines are explicit choices.
ENGINE_ORDER = ["xunpu", "audiveris", "oemer"]


def available_engines() -> list[Engine]:
    return [_REGISTRY[k] for k in ENGINE_ORDER]


def get_engine(key: str) -> Engine:
    return _REGISTRY[key]


def default_engine_key() -> str:
    for key in ENGINE_ORDER:
        if _REGISTRY[key].available():
            return key
    return "xunpu"
