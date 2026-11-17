"""Create a byte-identical source work copy from protected reference files."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references" / "SAC-RWB-main"
DESTINATION = ROOT / "src" / "sac_rwb"
REPORT = ROOT / "plan" / "p0" / "snapshot.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protected = json.loads(
        (ROOT / "plan" / "review" / "protected-integrity.json").read_text(encoding="utf-8")
    )
    entries = [
        item for item in protected["files"]
        if item["path"].startswith("references/SAC-RWB-main/")
    ]
    if DESTINATION.exists() and any(DESTINATION.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite nonempty work copy: {DESTINATION}")

    checked = []
    for item in entries:
        source = ROOT / item["path"]
        actual = digest(source)
        if actual != item["sha256"]:
            raise RuntimeError(f"Protected source differs from baseline: {source}")
        checked.append((source, item, actual))

    for source, _, actual in checked:
        target = DESTINATION / source.relative_to(REFERENCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        if digest(target) != actual:
            raise RuntimeError(f"Copy differs from source: {target}")

    route = ET.parse(REFERENCE / "real_data" / "newroute1220_start0.xml")
    network = ET.parse(REFERENCE / "real_data" / "CSeditClean_1.net_threelegs.xml")
    routes = Counter(element.tag for element in route.getroot())
    junction_types = {
        element.get("id"): element.get("type")
        for element in network.getroot().findall("junction")
    }
    report = {
        "scope": "P0 reference and work-copy inventory; no simulation run",
        "work_copy_file_count": len(checked),
        "copy_matches_reference": True,
        "files": [
            {
                "source": item["path"],
                "copy": (DESTINATION / source.relative_to(REFERENCE))
                .relative_to(ROOT).as_posix(),
                "sha256": actual,
                "bytes": source.stat().st_size,
            }
            for source, item, actual in checked
        ],
        "route_counts": dict(routes),
        "target_junction_types": {
            key: junction_types.get(key) for key in ("229", "499", "332", "334")
        },
        "requirements_overlay": "plan/p0/requirements-server.txt",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified and copied {len(checked)} files; wrote {REPORT}")


if __name__ == "__main__":
    main()
