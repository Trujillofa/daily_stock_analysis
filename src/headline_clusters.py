"""Group headlines that are the same story.

The groups are ids only. This module does not emit a buy or sell.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from src.embedding_rank import EmbedError, apply_prompt, cosine_similarity, embedder_from_env


def cluster_headlines(
    items: Sequence[dict[str, str]],
    embed: Callable[[str], Sequence[float]],
    *,
    threshold: float = 0.85,
) -> list[dict[str, list[str]]]:
    clusters: list[dict[str, object]] = []
    for item in items:
        title = item["title"].strip()
        if title == "":
            continue
        vector = list(embed(apply_prompt(title, "Document")))
        placed = False
        for cluster in clusters:
            centroid = cluster["vector"]
            assert isinstance(centroid, list)
            if cosine_similarity(vector, centroid) >= threshold:
                ids = cluster["ids"]
                assert isinstance(ids, list)
                ids.append(item["id"])
                placed = True
                break
        if not placed:
            clusters.append({"ids": [item["id"]], "vector": vector})
    return [{"ids": cluster["ids"]} for cluster in clusters]  # type: ignore[dict-item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cluster headlines. No trade signal.")
    parser.add_argument("headlines", type=Path, help="JSON list of {id, title}")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args(argv)
    payload = json.loads(args.headlines.read_text(encoding="utf-8"))
    try:
        groups = cluster_headlines(payload, embedder_from_env(), threshold=args.threshold)
    except EmbedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"groups": groups, "signal": None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
