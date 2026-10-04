import json
from xml.etree import ElementTree as ET

from xunpu_graph_omr.evaluation.portfolio import VARIANTS, build_portfolio


def test_portfolio_artifacts_and_clean_roundtrip(tmp_path):
    report = build_portfolio(tmp_path)
    assert report["summary"]["cases"] == len(VARIANTS) * 2
    assert json.loads((tmp_path / "results.json").read_text())["seed"] == 2026
    for case in report["cases"]:
        assert (tmp_path / case["image"]).is_file()
        if case["error"]:
            assert not case["exact"]
            continue
        document = json.loads((tmp_path / case["omr"]).read_text())
        assert document["source"] == case["image"]
        assert (tmp_path / case["overlay"]).is_file()
        root = ET.parse(tmp_path / case["xml"]).getroot()
        notes = root.findall(".//part/measure/note")
        assert len(notes) == case["actual"]
        if case["variant"] == "clean":
            assert case["exact"]
            assert case["measure_legal_rate"] == 1
            reconstructed = [
                n.findtext("pitch/step") + n.findtext("pitch/octave") for n in notes
            ]
            assert reconstructed == [n["pitch"] for n in case["notes"]]
            divisions = int(root.findtext(".//attributes/divisions"))
            assert [int(n.findtext("duration")) / divisions for n in notes] == [
                n["duration"] for n in case["notes"]
            ]
