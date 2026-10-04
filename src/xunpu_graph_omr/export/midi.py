"""Public MIDI export facade.

The implementation is dependency-free and delegates to ``midi_simple`` so the
CLI and desktop application share the same deterministic export path.
"""
from xunpu_graph_omr.export.midi_simple import write_midi, write_midi_notes

__all__ = ["write_midi", "write_midi_notes"]
