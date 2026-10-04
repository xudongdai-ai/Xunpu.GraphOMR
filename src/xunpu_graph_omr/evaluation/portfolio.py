"""Reproducible synthetic stress evaluation and a self-contained result viewer."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from xunpu_graph_omr.decoding.decoder import build_omr_document
from xunpu_graph_omr.vision.image_io import write_json
from xunpu_graph_omr.export.musicxml import write_musicxml
from xunpu_graph_omr.pipeline import recognize_path, write_debug_overlay
from xunpu_graph_omr.evaluation.synthetic import draw_duration_score, draw_simple_score

VARIANTS = ("clean", "scale_075", "scale_150", "rotate_minus2", "rotate_plus2",
            "blur", "jpeg_45", "noise_12")


def transform(gray: np.ndarray, variant: str) -> np.ndarray:
    if variant.startswith("scale_"):
        scale = {"scale_075": 0.75, "scale_150": 1.5}[variant]
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if variant.startswith("rotate_"):
        h, w = gray.shape
        angle = -2 if variant == "rotate_minus2" else 2
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(gray, matrix, (w, h), borderValue=255)
    if variant == "blur":
        return cv2.GaussianBlur(gray, (3, 3), 0.7)
    if variant == "jpeg_45":
        ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 45])
        if not ok:
            raise RuntimeError("JPEG encoder failed")
        return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if variant == "noise_12":
        rng = np.random.default_rng(2026)
        return np.clip(gray.astype(float) + rng.normal(0, 12, gray.shape), 0, 255).astype(np.uint8)
    if variant == "clean":
        return gray.copy()
    raise ValueError(f"Unknown variant: {variant}")


def build_portfolio(output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for fixture, draw in (("melody", draw_simple_score), ("durations", draw_duration_score)):
        base = output / f"{fixture}.png"
        truth = draw(base)
        write_json(output / f"{fixture}.truth.json", {**truth, "image": base.name})
        gray = cv2.imread(str(base), cv2.IMREAD_GRAYSCALE)
        for variant in VARIANTS:
            name = f"{fixture}-{variant}"
            image = output / f"{name}.png"
            if not cv2.imwrite(str(image), transform(gray, variant)):
                raise RuntimeError(f"Cannot write {image}")
            row = {"id": name, "variant": variant, "image": image.name,
                   "expected": len(truth["notes"]), "error": None}
            start = perf_counter()
            try:
                result = recognize_path(image)
                notes = sorted(result.notes, key=lambda n: (n.measure, n.staff_index, n.x))
                expected = truth["notes"]
                denominator = max(len(expected), len(notes), 1)
                pitch_matches = sum(t["pitch_name"] == n.pitch_name for t, n in zip(expected, notes))
                duration_matches = sum(t.get("duration", 1) == n.duration for t, n in zip(expected, notes))
                xml = output / f"{name}.musicxml"
                write_musicxml(result, xml, title=name)
                parsed = ET.parse(xml)
                exported = parsed.findall(".//part/measure/note")
                xml_valid = len(exported) == len(notes) and all(
                    float(n.findtext("duration", "0")) > 0 for n in exported
                )
                overlay = output / f"{name}.overlay.png"
                write_debug_overlay(result, overlay)
                document = build_omr_document(result)
                document["source"] = image.name
                write_json(output / f"{name}.omr.json", document)
                row.update({
                    "actual": len(notes), "pitch_accuracy": pitch_matches / denominator,
                    "duration_accuracy": duration_matches / denominator,
                    "exact": pitch_matches == duration_matches == len(expected) == len(notes),
                    "measure_legal_rate": sum(m.status == "complete" for m in result.measures)
                    / max(len(result.measures), 1),
                    "xml_parseable": xml_valid, "overlay": overlay.name, "xml": xml.name,
                    "omr": f"{name}.omr.json", "notes": [
                        {"pitch": n.pitch_name, "type": n.note_type, "duration": n.duration,
                         "measure": n.measure, "confidence": round(n.confidence, 3)} for n in notes
                    ], "node_types": result.graph.node_type_counts(),
                    "edge_types": result.graph.edge_type_counts(),
                    "steps": [{"name": s.name, "status": s.status, "detail": s.detail} for s in result.steps],
                })
            except Exception as exc:
                row.update({"error": str(exc), "actual": 0, "pitch_accuracy": 0,
                            "duration_accuracy": 0, "exact": False, "xml_parseable": False})
            row["elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
            cases.append(row)
    summary = {
        "cases": len(cases), "exact_cases": sum(c["exact"] for c in cases),
        "mean_pitch_accuracy": sum(c["pitch_accuracy"] for c in cases) / len(cases),
        "mean_duration_accuracy": sum(c["duration_accuracy"] for c in cases) / len(cases),
        "parseable_exports": sum(c["xml_parseable"] for c in cases),
    }
    payload = {"protocol": "synthetic-stress-v1", "seed": 2026,
               "scope": "Two procedural fixtures, eight variants each; not independent real-world scores.",
               "metric": "Index-aligned matches / max(expected count, detected count, 1); macro average across cases.",
               "export_check": "XML parse and positive note duration check, not MusicXML XSD validation.",
               "engine": "xunpu / classical detector / rule-based graph inference",
               "environment": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                               "numpy": np.__version__, "platform": platform.platform()},
               "summary": summary, "cases": cases}
    write_json(output / "results.json", payload)
    data = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    (output / "index.html").write_text(HTML.replace("__DATA__", data), encoding="utf-8")
    rows = ["# Synthetic Stress Evaluation", "", payload["scope"], "", payload["metric"], "",
            "| Case | Pitch | Duration | Legal measures | XML |",
            "| --- | ---: | ---: | ---: | --- |"]
    rows.extend(f"| {c['id']} | {c['pitch_accuracy']:.1%} | {c['duration_accuracy']:.1%} | "
                f"{c.get('measure_legal_rate', 0):.1%} | {c['xml_parseable']} |" for c in cases)
    rows += ["", "Limitations: no independent real-world evaluation, trained weights, or Audiveris superiority claim.",
             "Timing includes recognition and artifact export; it is not isolated inference latency.",
             "", "Reproduce: `xunpu-graph-omr portfolio --output out/evaluation`."]
    (output / "RESULTS.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return payload


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xunpu GraphOMR | Recognition Workbench</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f7f6;color:#192b26;font:14px system-ui,sans-serif;letter-spacing:0}
header{background:#fff;border-bottom:1px solid #d8e3de;padding:20px 28px;display:flex;justify-content:space-between;align-items:center;gap:20px}
h1{font-size:24px;margin:0}header p{margin:6px 0 0;color:#52645d}a{color:#176b4c}main{max-width:1300px;margin:auto;padding:24px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-bottom:1px solid #ccdcd3;margin-bottom:24px}
.metrics div{padding:8px 16px 20px 0}.metrics strong{display:block;font-size:26px}.metrics span{color:#52645d}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}select,button{font:inherit;padding:8px 12px;border:1px solid #b7ccc0;border-radius:5px;background:white;color:#192b26}button[aria-pressed=true]{background:#176b4c;color:white}
.score{width:100%;display:block;background:white;border:1px solid #d4e1da;object-fit:contain;max-height:420px}
.workspace{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:28px}h2{font-size:17px;margin:24px 0 12px}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:9px;border-bottom:1px solid #e1e9e5}th{font-weight:600;color:#52645d}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.7;padding:16px;background:#e8efeb}.scope{color:#52645d;line-height:1.6}.tablewrap{overflow-x:auto}
@media(max-width:800px){header{padding:16px;align-items:flex-start}main{padding:16px}.workspace{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:21px}}
</style>
<header><div><h1>Xunpu GraphOMR</h1><p>Recognition Workbench / Heterogeneous Graphs &amp; Musical Constraints</p></div><a href="results.json" download>Evaluation JSON</a></header>
<main><div class="metrics" id="metrics"></div>
<p class="scope">Independent classical OMR prototype. Two generated scores, eight image variants each. These results measure synthetic regression behavior; real-world accuracy is not established. Graph inference uses rules; HGT training code is experimental.</p>
<div class="toolbar"><label for="case">Score</label><select id="case"></select><button id="original" aria-pressed="false">Original</button><button id="overlay" aria-pressed="true">Detections</button><a id="xml" download>MusicXML</a><a id="omr" download>OMR JSON</a></div>
<img id="score" class="score" alt="Selected score recognition overlay">
<div class="workspace"><section><h2>Recovered Notes</h2><p id="status"></p><div class="tablewrap"><table><thead><tr><th>Pitch</th><th>Type</th><th>Beats</th><th>Measure</th><th>Confidence</th></tr></thead><tbody id="notes"></tbody></table></div></section>
<section><h2>Graph Relations</h2><pre id="graph"></pre><h2>Processing Trace</h2><pre id="steps"></pre></section></div>
<h2>All Evaluation Cases</h2><div class="tablewrap"><table><thead><tr><th>Case</th><th>Pitch</th><th>Duration</th><th>Exact</th><th>XML Parseable</th></tr></thead><tbody id="results"></tbody></table></div>
<p class="scope">Index-aligned pitch/duration matches, normalized by the larger note count. Variants share source fixtures. Confidence is a heuristic, not a calibrated probability. XML parseability is not XSD conformance. <a href="RESULTS.md">Protocol and limitations</a>.</p></main>
<script>
const data=__DATA__,s=data.summary,percent=v=>(v*100).toFixed(1)+'%';
document.querySelector('#metrics').innerHTML=[[s.cases,'Synthetic cases'],[percent(s.mean_pitch_accuracy),'Mean pitch match'],[percent(s.mean_duration_accuracy),'Mean duration match'],[s.exact_cases+'/'+s.cases,'Exact note sequences']].map(([v,k])=>'<div><strong>'+v+'</strong><span>'+k+'</span></div>').join('');
const select=document.querySelector('#case');data.cases.forEach(c=>select.add(new Option(c.id,c.id)));
const results=document.querySelector('#results');data.cases.forEach(c=>{const tr=results.insertRow();[c.id,percent(c.pitch_accuracy),percent(c.duration_accuracy),c.exact?'Yes':'No',c.xml_parseable?'Yes':'No'].forEach(v=>tr.insertCell().textContent=v)});
let overlay=true;
function show(){const c=data.cases.find(c=>c.id===select.value);document.querySelector('#score').src=overlay&&c.overlay?c.overlay:c.image;document.querySelector('#score').alt=c.id+(overlay?' detections':' original');
document.querySelector('#status').textContent=c.error||`${c.actual}/${c.expected} notes | Pitch ${percent(c.pitch_accuracy)} | Duration ${percent(c.duration_accuracy)} | ${c.elapsed_ms} ms including exports`;
for(const key of ['xml','omr']){const a=document.querySelector('#'+key);a.hidden=!c[key];if(c[key])a.href=c[key]}
const tbody=document.querySelector('#notes');tbody.replaceChildren();(c.notes||[]).forEach(n=>{const tr=tbody.insertRow();[n.pitch,n.type,n.duration,n.measure,n.confidence].forEach(v=>tr.insertCell().textContent=v)});
document.querySelector('#graph').textContent=JSON.stringify({nodes:c.node_types||{},edges:c.edge_types||{}},null,2);document.querySelector('#steps').textContent=(c.steps||[]).map(t=>t.name+' ['+t.status+'] '+t.detail).join('\n');
document.querySelector('#original').setAttribute('aria-pressed',String(!overlay));document.querySelector('#overlay').setAttribute('aria-pressed',String(overlay));}
select.onchange=show;document.querySelector('#original').onclick=()=>{overlay=false;show()};document.querySelector('#overlay').onclick=()=>{overlay=true;show()};show();
</script></html>'''
