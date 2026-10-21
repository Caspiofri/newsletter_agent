"""Stable train/test split of the ranking golden set (60 train / 20 test of 80 labels, i.e. 25 percent test).

The split is keyed on the normalized title plus the source, not the URL, so the same article labeled under two
addresses (a Substack redirect and the real post) always lands on the same side.
"""
import hashlib

TEST_PERCENT = 25
SPLITS = ("train", "test", "all")


def split_key(row: dict) -> str:
    title = " ".join((row.get("title") or "").split()).lower()
    return f"{title}|{(row.get('origin') or '').strip().lower()}"


def assign_split(row: dict, test_percent: int = TEST_PERCENT) -> str:
    bucket = int(hashlib.sha1(split_key(row).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < test_percent else "train"


def split_rows(rows: list[dict], which: str) -> list[dict]:
    if which not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {which!r}")
    return list(rows) if which == "all" else [r for r in rows if assign_split(r) == which]


def split_counts(rows: list[dict]) -> dict:
    sides = [assign_split(r) for r in rows]
    return {"train": sides.count("train"), "test": sides.count("test")}
