"""Merge the JUnit reports of one job's pytest phases into one report (#421 rule 5): `merge_junit.py OUT PART...`
writes OUT as a `<testsuites>` holding every `<testsuite>` of every PART, in the order given. A PART that is
missing (its phase died before writing it) shows in OUT as one failed test case naming it, so a reader that
judges from the report alone sees red; one that cannot be parsed stops the merge with an error, so a half-written report is never
passed as a whole one. OUT may be one of the PARTs."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def merge(out: Path, parts: list[Path]) -> int:
    merged = ET.Element("testsuites")
    for part in parts:
        if not part.exists():
            suite = ET.SubElement(merged, "testsuite", name="missing report", tests="1", failures="1")
            case = ET.SubElement(suite, "testcase", classname="merge_junit", name=f"report of the phase missing: {part.name}")
            ET.SubElement(case, "failure", message=f"{part.name} was not written: the phase ended before its report")
            continue
        root = ET.parse(part).getroot()
        merged.extend([root] if root.tag == "testsuite" else list(root))
    ET.ElementTree(merged).write(out, encoding="utf-8", xml_declaration=True)
    return sum(len(list(suite.iter("testcase"))) for suite in merged)


if __name__ == "__main__":
    print(f"merged report: {merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])} test cases")
