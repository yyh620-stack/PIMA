"""Editable bus-cluster configuration for partitioned DNN training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


DEFAULT_CLUSTER_FILES = {
    "case118": Path(__file__).with_name("case118_clusters.json"),
}


def load_bus_clusters(
    case_name: str,
    nbus: int,
    cluster_file: Optional[str] = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Load and validate a complete, non-overlapping 1-based bus partition."""
    if cluster_file is None:
        try:
            path = DEFAULT_CLUSTER_FILES[case_name]
        except KeyError as exc:
            raise ValueError(
                f"No default cluster file for {case_name}; pass --cluster-file"
            ) from exc
    else:
        path = Path(cluster_file)

    raw = json.loads(path.read_text(encoding="utf-8"))
    cluster_values = raw.get("clusters") if isinstance(raw, dict) else raw
    if not isinstance(cluster_values, list) or not cluster_values:
        raise ValueError(f"{path}: 'clusters' must be a non-empty list")

    clusters: List[Tuple[int, ...]] = []
    for number, values in enumerate(cluster_values, start=1):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: cluster {number} must be a non-empty list")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"{path}: cluster {number} contains a non-integer bus")
        clusters.append(tuple(values))

    _validate_clusters(clusters, nbus, path)
    return tuple(clusters)


def _validate_clusters(
    clusters: Sequence[Sequence[int]], nbus: int, path: Path
) -> None:
    flattened = [bus for cluster in clusters for bus in cluster]
    out_of_range = sorted({bus for bus in flattened if bus < 1 or bus > nbus})
    if out_of_range:
        raise ValueError(f"{path}: bus numbers out of range: {out_of_range}")

    counts = {bus: flattened.count(bus) for bus in set(flattened)}
    duplicates = sorted(bus for bus, count in counts.items() if count > 1)
    missing = sorted(set(range(1, nbus + 1)) - set(flattened))
    if duplicates or missing:
        raise ValueError(
            f"{path}: clusters must partition all buses exactly once; "
            f"duplicates={duplicates}, missing={missing}"
        )
