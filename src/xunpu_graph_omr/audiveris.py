from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_AUDIVERIS = Path("/Applications/Audiveris.app/Contents/MacOS/Audiveris")

_REL_MACOS = Path("Contents/MacOS/Audiveris")


def _bundled_candidates() -> list[Path]:
    """Locations of an Audiveris copy shipped *inside* our own app/package."""
    candidates: list[Path] = []
    # PyInstaller frozen app: XunpuGraphOMR.app/Contents/MacOS/XunpuGraphOMR
    #   -> bundled Audiveris at XunpuGraphOMR.app/Contents/Resources/Audiveris.app
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent          # Contents/MacOS
        candidates.append(exe_dir.parent / "Resources" / "Audiveris.app" / _REL_MACOS)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "Audiveris.app" / _REL_MACOS)
    # source checkout: vendor/Audiveris.app next to the repo
    candidates.append(Path(__file__).resolve().parents[2] / "vendor" / "Audiveris.app" / _REL_MACOS)
    return candidates


def find_audiveris(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    env = os.environ.get("OWN_OMR_AUDIVERIS")
    if env and Path(env).exists():
        return Path(env)
    for cand in _bundled_candidates():
        if cand.exists():
            return cand
    if DEFAULT_AUDIVERIS.exists():
        return DEFAULT_AUDIVERIS
    found = shutil.which("audiveris")
    return Path(found) if found else None


def run_audiveris_export(
    image_path: str | Path,
    output_dir: str | Path,
    executable: str | Path | None = None,
    timeout_seconds: int = 180,
) -> Path:
    audiveris = find_audiveris(executable)
    if audiveris is None:
        raise FileNotFoundError("Audiveris executable not found")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(audiveris),
        "-batch",
        "-transcribe",
        "-export",
        "-output",
        str(output),
        str(image_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Audiveris failed with code {completed.returncode}:\n{completed.stdout[-4000:]}")

    candidates = sorted(
        list(output.rglob("*.musicxml"))
        + list(output.rglob("*.xml"))
        + list(output.rglob("*.mxl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Audiveris completed but no MusicXML output was found under {output}")
    return candidates[0]
