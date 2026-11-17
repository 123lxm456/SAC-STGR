"""Set A4 and 20 mm margins on a Pandoc-generated chapter DOCX.

Usage: python scripts/format_chapter_docx.py path/to/chapter.docx
Uses only the Python standard library; does not change text or equations.
"""

from pathlib import Path
from io import BytesIO
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)


def main(path: Path) -> None:
    with ZipFile(path, "r") as source:
        members = [(item, source.read(item.filename)) for item in source.infolist()]
    output = []
    found = False
    for item, payload in members:
        if item.filename == "word/document.xml":
            for _, namespace in ET.iterparse(BytesIO(payload), events=("start-ns",)):
                prefix, uri = namespace
                try:
                    ET.register_namespace(prefix or "", uri)
                except ValueError:
                    pass
            root = ET.fromstring(payload)
            for section in root.iter(f"{{{W}}}sectPr"):
                found = True
                size = section.find(f"{{{W}}}pgSz")
                if size is None:
                    size = ET.SubElement(section, f"{{{W}}}pgSz")
                size.set(f"{{{W}}}w", "11906")
                size.set(f"{{{W}}}h", "16838")
                margin = section.find(f"{{{W}}}pgMar")
                if margin is None:
                    margin = ET.SubElement(section, f"{{{W}}}pgMar")
                for side in ("top", "bottom", "left", "right"):
                    margin.set(f"{{{W}}}{side}", "1134")
            payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        output.append((item, payload))
    if not found:
        raise ValueError("DOCX has no section properties")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".docx", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with ZipFile(temp_path, "w") as target:
            for item, payload in output:
                target.writestr(item, payload)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
