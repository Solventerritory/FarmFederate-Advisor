"""Demo driver for :class:`FarmMemoryAgent`.

Stores a handful of synthetic Crop Stress Analysis Reports for a single demo farm
in an in-memory Qdrant instance, runs a per-farm visual similarity query, and
exports the farm's history to CSV. Used by the ``Qdrant Memory Smoke`` workflow
to produce ``results/demo_farm_1_history.csv``.

Usage:
    python -m backend.demo_farm_memory_demo
    python -m backend.demo_farm_memory_demo --out results/demo_farm_1_history.csv
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np

from backend.farm_memory_agent import FarmMemoryAgent

FARM_ID = "demo-farm-1"
DEFAULT_OUT = os.path.join("results", "demo_farm_1_history.csv")

# (report text, severity) pairs used to seed the demo farm history.
DEMO_REPORTS: List[Tuple[str, str]] = [
    ("Early blight lesions on lower leaves; irrigation adequate.", "low"),
    ("Nitrogen deficiency suspected; chlorosis spreading upward.", "medium"),
    ("Water stress with leaf curl across the eastern plot.", "high"),
]


def _embedding(dim: int, seed: int) -> List[float]:
    """Deterministic unit-norm embedding so demo runs are reproducible."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    return list(vec / np.linalg.norm(vec))


def run_demo(out_path: str = DEFAULT_OUT, qdrant_url: str = ":memory:") -> str:
    """Seed the demo farm, query it, and export the history CSV.

    Returns:
        Path of the exported CSV.
    """
    agent = FarmMemoryAgent(qdrant_url=qdrant_url)
    agent.init_collection(recreate=True)

    for i, (report, severity) in enumerate(DEMO_REPORTS):
        pid = agent.store_report(
            report,
            _embedding(agent._visual_dim, seed=i),
            _embedding(agent._semantic_dim, seed=100 + i),
            farm_id=FARM_ID,
            metadata={"severity": severity},
        )
        print(f"Stored report {i + 1}/{len(DEMO_REPORTS)} id={pid} severity={severity}")

    # Query with the first report's visual embedding: memory is scoped to FARM_ID.
    hits = agent.retrieve_similar_by_image(_embedding(agent._visual_dim, seed=0), farm_id=FARM_ID, top_k=3)
    print(f"Retrieved {len(hits)} similar historical reports for {FARM_ID}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    path = agent.export_reports_to_csv(FARM_ID, out_path=out_path)
    print(f"Exported farm history to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="FarmMemoryAgent demo")
    parser.add_argument("--out", default=DEFAULT_OUT, help="path for the exported history CSV")
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", ":memory:"),
        help="Qdrant URL, or ':memory:' for an in-memory demo instance",
    )
    args = parser.parse_args()
    run_demo(out_path=args.out, qdrant_url=args.qdrant_url)


if __name__ == "__main__":
    main()
