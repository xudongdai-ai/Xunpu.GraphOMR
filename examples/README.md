# Examples

`input/` contains small procedural fixtures and truth JSON used for public regression checks.

`output/` contains representative clean recognition artifacts plus the committed synthetic evaluation summary. The complete 16-case workbench is generated on demand rather than stored as a large set of repetitive files:

```bash
xunpu-graph-omr portfolio --output out/evaluation
```

Representative artifacts include an overlay, structured OMR JSON, and MusicXML export.
