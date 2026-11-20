"""Static checks for the proposed spatial and temporal Transformer design.

This script checks references/ when the local snapshot is present and checks
that output/ is absent. It writes only the report under plan/.
It does not execute SUMO, train a model, or establish performance claims.
"""

from pathlib import Path
import hashlib
import json
import re
import argparse


ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append({"name": name, "passed": bool(passed), "detail": detail})


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def section_34(text):
    match = re.search(r"(?s)## 3\.4[^\r\n]*.*\Z", text)
    return match.group(0) if match else None


manifest = json.loads(read("plan/review/protected-integrity.json"))
expected = {item["path"]: item["sha256"] for item in manifest["files"]
            if item["path"].startswith("references/")}
reference_dir = ROOT / "references"
if reference_dir.exists():
    actual = {}
    for file in reference_dir.rglob("*"):
        if file.is_file():
            rel = file.relative_to(ROOT).as_posix()
            actual[rel] = digest(file.read_bytes())
    changed = [name for name in sorted(set(expected) | set(actual))
               if expected.get(name) != actual.get(name)]
    check("protected references inventory and SHA-256 unchanged",
          not changed, changed)
else:
    check("local references snapshot omitted from public repository", True)
check("output directory removed by request", not (ROOT / "output").exists())

chapters = {
    "full": read("chapters/03_methods_experiments.md"),
    "brief": read("chapters/03_methods_experiments_brief.md"),
}
for name, body in chapters.items():
    original = manifest["chapter34" if name == "full" else "brief34"]
    frozen = section_34(body)
    check(f"{name} chapter 3.4 unchanged",
          frozen is not None and digest(frozen.encode("utf-8")) == original)
    check(f"{name} chapter ends at 3.4",
          "## 3.5 " not in body and "空间 Transformer" in body
          and "时间 Transformer" in body)
    check(f"{name} preserves core formula and limitations",
          all(term in body for term in
              ("[10,9,64]", "风险分数", "组", "Sigmoid", "近期偏置")))

spec = read("research_implementation_plan.md")
readme = read("README.md")
stage = read("plan/stage-gates.md")
trace = read("plan/review/method-experiment-traceability.md")
check("technical specification has connected spatiotemporal pipeline",
      all(term in spec for term in
          ("[B,10,9,64]", "Encoder}_S", "Encoder}_T",
           "末步读出", "有效步掩码", "M1", "C1", "C2", "C3")))
acceptance = json.loads(read('plan/acceptance/p012-accepted-20261120.json'))
check("README and stage identify phases backed by P012 acceptance record",
      all('P{}'.format(i) in item for i in range(6) for item in (readme, stage))
      and acceptance.get('accepted') is True
      and all(acceptance.get('stages', {}).get('P{}'.format(i)) == 'passed' for i in range(3)))
check("method claims map to experiments",
      all(term in trace for term in
          ("M1", "B0", "C1", "C2", "C3", "A-B", "校准")))
check("required project planning files",
      all((ROOT / rel).is_file() for rel in
          ("plan/project-overview.md", "plan/outline.md",
           "plan/task-packets/spatiotemporal-redesign.md",
           "plan/evidence-map.md")))

doc_files = [
    ROOT / rel for rel in
    ("README.md", "research_implementation_plan.md",
     "chapters/03_methods_experiments.md",
     "chapters/03_methods_experiments_brief.md",
     "plan/project-overview.md", "plan/outline.md",
     "plan/stage-gates.md", "plan/progress.md",
     "plan/review/method-experiment-traceability.md")
]
broken = []
pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
for page in doc_files:
    for target in pattern.findall(page.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local = target.split("#", 1)[0]
        if local and not (page.parent / local).is_file():
            broken.append(f"{page.relative_to(ROOT)} -> {target}")
check("active local links resolve", not broken, broken)

report = {
    "scope": "Static documents, immutable chapter 3.4 and protected file hashes only",
    "passed": all(item["passed"] for item in CHECKS),
    "checks": CHECKS,
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, default=ROOT / 'plan/review/plan-verification.json')
out = parser.parse_args().report
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["passed"] else 1)
