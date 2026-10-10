# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unittest

from src.headline_clusters import cluster_headlines


def lexical_embed(text: str) -> list[float]:
    vector = [0.0] * 256
    for token in re.findall(r"[0-9a-z]+", text.casefold()):
        bucket = 0
        for char in token:
            bucket = (bucket * 33 + ord(char)) % 256
        vector[bucket] += 1.0
    return vector


class HeadlineClusterTests(unittest.TestCase):
    def test_same_story_groups_together_and_there_is_no_trade(self) -> None:
        items = [
            {"id": "a", "title": "Copper miner raises full year output guidance"},
            {"id": "b", "title": "Copper miner raises full year output guidance again"},
            {"id": "c", "title": "Airline cuts winter flight schedule"},
        ]
        groups = cluster_headlines(items, lexical_embed, threshold=0.5)
        flat = [set(group["ids"]) for group in groups]
        self.assertTrue(any({"a", "b"} <= group for group in flat))
        self.assertTrue(all("signal" not in group and "side" not in group for group in groups))


if __name__ == "__main__":
    unittest.main()
