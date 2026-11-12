"""Tiny on-disk cache so experiments never re-score the same file twice."""
from __future__ import annotations

import hashlib
import json
import os

from .types import FaceReading


def file_key(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CachedScorer:
    def __init__(self, scorer, cache_path: str = "outputs/score_cache.json"):
        self.scorer, self.path = scorer, cache_path
        self.name = scorer.name
        self._data = {}
        if os.path.exists(cache_path):
            try:
                self._data = json.load(open(cache_path))
            except Exception:                  # noqa: BLE001
                self._data = {}

    def read(self, image) -> FaceReading:
        if isinstance(image, (str, os.PathLike)):
            k = self.name + getattr(self.scorer, "version", "") + ":" + file_key(str(image))
            if k in self._data:
                return FaceReading.from_dict(self._data[k])
            r = self.scorer.read(image)
            self._data[k] = r.to_dict()
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            json.dump(self._data, open(self.path, "w"))
            return r
        return self.scorer.read(image)
