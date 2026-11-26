# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit user matrices, bounded expansion, and stable ordering."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
from .catalog import generate
from .model import Case


def load_cases(path: Path):
    if path.suffix == ".jsonl":
        from .runner import read_jsonl
        yield from (Case.from_dict(item) for item in read_jsonl(path))
        return
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1: raise ValueError("matrix schema_version must be 1")
    limit = spec.get("max_cases", 1000000)
    if not isinstance(limit, int) or not 1 <= limit <= 10000000: raise ValueError("invalid max_cases")
    seen = set()
    def unique(item):
        if item.id in seen: return False
        seen.add(item.id)
        if len(seen) > limit: raise ValueError("matrix exceeds max_cases")
        return True
    if spec.get("base_profile"):
        for item in generate(spec["base_profile"]):
            if unique(item): yield item
    for raw in spec.get("cases", []):
        item = Case.from_dict(raw)
        if unique(item): yield item
    for sweep in spec.get("sweeps", []):
        axes = sweep.get("axes", {})
        if any(not isinstance(values, list) or not values for values in axes.values()):
            raise ValueError("each sweep axis must be a nonempty list")
        names = list(axes)
        for values in itertools.product(*(axes[name] for name in names)):
            params = {**sweep.get("params", {}), **dict(zip(names, values))}
            item = Case(sweep["family"], params, tuple(sweep.get("tags", ())))
            if unique(item): yield item
