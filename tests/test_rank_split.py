"""Offline tests for the stable train/test split of the ranking labels. Run: python -m tests.test_rank_split"""
from evals import rank_split


def _row(title, origin="inbox", url="https://x.example/a"):
    return {"title": title, "origin": origin, "url": url, "label": "must"}


def test_assign_split_is_stable_across_runs():
    row = _row("Attention Is All You Need")
    assert rank_split.assign_split(row) == rank_split.assign_split(dict(row))
    # pinned: sha1("attention is all you need|inbox")[:8] % 100 == 50, and "paper 5|hackernews" -> 3
    assert rank_split.assign_split(row) == "train"
    assert rank_split.assign_split(_row("Paper 5", "hackernews")) == "test"


def test_same_article_under_two_addresses_same_split():
    for i in range(50):
        a = _row(f"Story {i}", url="https://substack.com/redirect/2/abc")
        b = _row(f"Story {i}", url="https://real.example/story")
        assert rank_split.assign_split(a) == rank_split.assign_split(b)


def test_split_proportion_is_roughly_25_percent():
    rows = [_row(f"Synthetic title number {i}", origin=("inbox", "medium", "hackernews")[i % 3]) for i in range(1000)]
    test = sum(rank_split.assign_split(r) == "test" for r in rows)
    assert 200 <= test <= 300, test


def test_split_rows_all_train_test_partition():
    rows = [_row(f"Title {i}") for i in range(40)]
    train, test = rank_split.split_rows(rows, "train"), rank_split.split_rows(rows, "test")
    assert len(train) + len(test) == len(rows)
    assert not {id(r) for r in train} & {id(r) for r in test}
    assert rank_split.split_rows(rows, "all") == rows
    assert rank_split.split_counts(rows) == {"train": len(train), "test": len(test)}


def test_split_key_ignores_case_and_whitespace():
    assert rank_split.split_key(_row("  Hello   World ")) == rank_split.split_key(_row("hello world"))
    assert rank_split.split_key(_row("Hello", "inbox")) != rank_split.split_key(_row("Hello", "medium"))
    assert rank_split.split_key(_row("Hello", "inbox")) == "hello|inbox"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
