"""HuggingFace-mirror readers and the row helpers the loaders share.

The readers come in two shapes and the difference matters for this box: :func:`_head` returns the
first N rows (or the first N of a seeded shuffle) **without materialising the file**, and
:func:`_parquet` returns all of it.  A loader that needs every row -- to group by comment id, or to
build the option list out of the label set, or to filter by a question id -- has to use the second
one; everything else should use the first.
"""
from __future__ import annotations

import gzip
import json
import random

from ..decision import DecisionExample, Question


HF_MIRROR = "https://hf-mirror.com"


def _local(repo: str, path: str) -> str:
    """The local path of one file in a dataset repo, fetched through the mirror if it is not here.

    One downloader for every reader, which each used to carry the same four lines -- and a fourth
    copy was about to be written for ``clinc_oos``.  ``HF_ENDPOINT`` is set as well as passed
    because ``huggingface_hub`` reads the environment for its own internal calls.
    """
    import os

    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset", endpoint=HF_MIRROR)


def _line_stream(local: str):
    """Yield the non-blank lines of a plain or gzipped text file, one at a time."""
    opener = gzip.open if local.endswith(".gz") else open
    with opener(local, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def _row_count(local: str) -> int:
    """How many rows the file holds, without reading it into memory."""
    if local.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.ParquetFile(local).metadata.num_rows
    return sum(1 for _ in _line_stream(local))


def _rows_at(local: str, slots: dict[int, int]) -> list[dict]:
    """The rows whose index is a key of ``slots``, each written to its own position.

    Row indices are the file's rows in order: what ``to_pylist()`` and a line list used to index
    into.  Stops as soon as every slot is filled, which is what makes the prefix case cheap -- a
    392,702-row file read for 400 rows never reaches the second row group.
    """
    out: list[dict | None] = [None] * len(slots)
    remaining = len(slots)
    if local.endswith(".parquet"):
        import pyarrow.parquet as pq

        batches = pq.ParquetFile(local).iter_batches(batch_size=4096)
        i = 0
        for batch in batches:
            for row in batch.to_pylist():
                if i in slots:
                    out[slots[i]] = row
                    remaining -= 1
                    if not remaining:
                        return [r for r in out if r is not None]
                i += 1
    else:
        i = -1
        for line in _line_stream(local):
            i += 1
            if i in slots:
                out[slots[i]] = json.loads(line)
                remaining -= 1
                if not remaining:
                    break
    return [r for r in out if r is not None]


def _head(repo: str, path: str, limit: int, seed: int | None = None) -> list[dict]:
    """The first ``limit`` rows of a file -- or of a seeded shuffle of it -- without holding it.

    This replaces ``rows = read_everything(); shuffle(rows); rows[:limit]``, which materialises the
    whole corpus to use a few thousand rows: amazon's per-language train file is 200,000 rows for
    3,500 used, xnli's is 392,702 for 400, ag_news' 120,000 for 3,500.  Those allocations are what
    repeatedly took this box's commit charge to the wall -- the same split segfaulting while an
    identical extraction of it had succeeded earlier.

    Shuffling *indices* is not an approximation.  ``random.shuffle`` applies the same permutation
    to a list of length n whatever its elements, so ``[rows[i] for i in order[:limit]]`` is exactly
    ``shuffle(rows)[:limit]`` for the same seed -- which is why this could replace the old form
    without moving a single cached row.  The replacement was checked that way, loader by loader,
    against the old expression.
    """
    local = _local(repo, path)
    n = _row_count(local)
    if seed is None:
        order = list(range(min(limit, n)))
    else:
        order = list(range(n))
        random.Random(seed).shuffle(order)
        order = order[:limit]
    return _rows_at(local, {idx: pos for pos, idx in enumerate(order)})


def _jsonl(repo: str, path: str) -> list[dict]:
    """Every row of a line-delimited JSON file, plain or gzipped.

    For loaders that genuinely need all the rows -- ``massive`` and ``sst5`` build their option
    list out of the label set of the whole file, and ``go_emotions`` groups rows by comment id.
    Everything else should use :func:`_head`.
    """
    return [json.loads(line) for line in _line_stream(_local(repo, path))]


def _parquet(repo: str, path: str) -> list[dict]:
    """Every row of a parquet file.  Only for loaders that genuinely need all of them."""
    import pyarrow.parquet as pq

    return pq.read_table(_local(repo, path)).to_pylist()


SPLIT_SEED = 20260922


def _by_question(rows, text_of, label_of, question: Question) -> list[DecisionExample]:
    index = {o: i for i, o in enumerate(question.options)}
    return [
        DecisionExample(
            state=text_of(r),
            question=question,
            answer=index[label_of(r)],
            true_probs=None,
            meta={"source": question.name},
        )
        for r in rows
        if label_of(r) in index and len(text_of(r).strip()) > 3
    ]
