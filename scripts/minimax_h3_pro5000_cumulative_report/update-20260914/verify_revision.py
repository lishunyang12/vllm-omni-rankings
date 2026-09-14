#!/usr/bin/env python3
"""Check the published report's numerical, archive, PDF and local-link contracts."""
import hashlib
import json
import statistics
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
REPO = REPORT.parent.parent
PDF = REPORT / "minimax_h3_653s_to_14s_cumulative_report.pdf"

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
    def handle_starttag(self, tag, attrs):
        self.targets.extend(v for k, v in attrs if k in ("href", "src") and v)

def main():
    d = json.loads((HERE / "data.json").read_text())
    assert len(d["revised_steps"]) == 14
    for step in d["revised_steps"]:
        delta = step["baseline_s"] - step["candidate_s"]
        assert abs(delta - step["saved_s"]) < 1e-9
        assert delta > 0.100
    samples = d["current"]["formal_s"]
    assert len(samples) == 5 and all(v < 15 for v in samples)
    assert abs(statistics.mean(samples) - 14.9692) < 1e-9
    assert abs(statistics.stdev(samples) - d["current"]["sample_sd_s"]) < 1e-9
    assert all(g["byteexact"] for g in d["current"]["video_byte_gates"])
    assert len(d["current"]["video_byte_gates"]) == 6
    archive = REPORT / "archive-20260907"
    original_sha = hashlib.sha256((archive / PDF.name).read_bytes()).hexdigest()
    assert original_sha == "a7d1fe30a1fad2688d2f54272250099950944112f4715abe84922e800293fa04"
    assert len(PdfReader(archive / PDF.name).pages) == 71
    assert hashlib.sha256((archive / "report_data.json").read_bytes()).hexdigest() == d["historical"]["source_sha256"]
    reader = PdfReader(PDF)
    assert len(reader.pages) == 24
    texts = [p.extract_text() for p in reader.pages]
    assert all(len(t.strip()) > 100 for t in texts)
    for value in ["27.4937", "14.9692", "0.7773", "0.3614", "0.5034", "0.97892", "43.0414"]:
        assert value in "\n".join(texts[:12])
    assert "D6 final" not in "\n".join(texts)
    assert "14.9300" not in "\n".join(texts)
    checked = []
    for html in [REPORT / "index.html", HERE / "index.html", archive / "index.html"]:
        parser = Links()
        parser.feed(html.read_text())
        for href in parser.targets:
            url = urlsplit(href)
            if url.scheme or not url.path:
                continue
            path = (html.parent / unquote(url.path)).resolve()
            assert path.exists(), (html, href)
            checked.append(str(path.relative_to(REPO)))
    pdf_links = 0
    public = "https://lishunyang12.github.io/vllm-omni-rankings/"
    for page in reader.pages:
        for ref in page.get("/Annots", []):
            action = ref.get_object().get("/A")
            if not action:
                continue
            uri = str(action.get("/URI", ""))
            if uri.startswith(public):
                path = REPO / unquote(urlsplit(uri[len(public):]).path)
                assert path.exists(), uri
                pdf_links += 1
    snapshots = json.loads((HERE / "evidence/source-manifest.json").read_text())
    for source in snapshots:
        assert hashlib.sha256((HERE / "evidence" / source["file"]).read_bytes()).hexdigest() == source["sha256"]
    result = {
        "status": "PASS", "pdf_pages": 24, "archived_pdf_pages": 71,
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "original_pdf_sha256_unchanged": original_sha,
        "checked_gain_rows": 14, "all_gains_gt_0_100s": True,
        "formal_mean_s": statistics.mean(samples), "all_five_under_15s": True,
        "html_local_links_checked": len(checked), "pdf_public_links_resolved_locally": pdf_links,
        "source_snapshots_verified": len(snapshots),
        "scope": "Document/artifact checks only. All 24 PDF pages were separately rendered for visual review. No GPU benchmark run."
    }
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
