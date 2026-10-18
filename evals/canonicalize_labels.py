"""Rewrite the URLs of an exported labels file to their canonical article addresses.

Usage: python -m evals.canonicalize_labels <exported labels json>
Writes <stem>.canonical.json next to the input. Makes network requests (follows each redirect) when run.
"""
import json
import sys
from pathlib import Path

from modules import link_utils


def canonicalize(exported: dict, resolver=link_utils.resolve_link) -> tuple[dict, int]:
    """Return (copy of exported with every label url canonical, number of urls changed). Input is not mutated."""
    out = {**exported, "labels": []}
    changed = 0
    for entry in exported.get("labels", []):
        url = entry.get("url")
        new = link_utils.canonical_url(url, resolver(url)[1]) if url else url
        changed += new != url
        out["labels"].append({**entry, "url": new} if "url" in entry else dict(entry))
    return out, changed


def main(path: str) -> None:
    src = Path(path)
    exported = json.loads(src.read_text(encoding="utf-8"))
    out, changed = canonicalize(exported)
    dest = src.with_name(f"{src.stem}.canonical.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{changed} of {len(out['labels'])} urls canonicalised -> {dest}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
