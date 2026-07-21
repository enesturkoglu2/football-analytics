#!/usr/bin/env python3
"""CLI entry point for Stage 4B manually approved ReID linking.

Src-layout bootstrap adds ``<project_root>/src`` so ``football_analytics``
imports resolve when the script is run directly.

Cosine values are read from candidate_pairs.jsonl for audit only.
No automatic threshold acceptance is performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if not _SRC_DIR.is_dir():
    print(f"error: expected src directory at {_SRC_DIR}", file=sys.stderr)
    sys.exit(2)
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from football_analytics.reid.linking import (  # noqa: E402
    DEFAULT_POLICY,
    LinkingError,
    run_link_reid_tracks,
)
from football_analytics.reid.writers import ReIDWritersError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    default_policy = _PROJECT_ROOT / DEFAULT_POLICY
    parser = argparse.ArgumentParser(
        description=(
            "Link ReID tracks using explicit manual approvals only. "
            "No cosine threshold and no uncontrolled Union-Find chaining."
        )
    )
    parser.add_argument(
        "--candidate-pairs",
        required=True,
        help="Path to candidate_pairs.jsonl",
    )
    parser.add_argument(
        "--track-embeddings-index",
        required=True,
        help="Path to track_embeddings.jsonl",
    )
    parser.add_argument(
        "--tracks",
        required=True,
        help="Path to Stage 3 tracks.jsonl",
    )
    parser.add_argument(
        "--manual-decisions",
        required=True,
        help="Path to manual pair decisions JSONL (may be empty)",
    )
    parser.add_argument(
        "--policy",
        default=str(default_policy),
        help=f"Linking policy YAML (default: {default_policy})",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for accepted edges, audit, global map, summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory atomically.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_link_reid_tracks(
            candidate_pairs=args.candidate_pairs,
            track_embeddings_index=args.track_embeddings_index,
            tracks=args.tracks,
            manual_decisions=args.manual_decisions,
            policy=args.policy,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (LinkingError, ReIDWritersError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = result["summary"]
    print(f"wrote {result['accepted_edges_path']}")
    print(f"wrote {result['audit_path']}")
    print(f"wrote {result['global_map_path']}")
    print(f"wrote {result['summary_path']}")
    print(
        "status=ok "
        f"raw_track_count={summary['raw_track_count']} "
        f"embedded_track_count={summary['embedded_track_count']} "
        f"candidate_pair_count={summary['candidate_pair_count']} "
        f"manual_decision_count={summary['manual_decision_count']} "
        f"requested_approved_edge_count={summary['requested_approved_edge_count']} "
        f"applied_accepted_edge_count={summary['applied_accepted_edge_count']} "
        f"linked_component_count={summary['linked_component_count']} "
        f"linked_track_count={summary['linked_track_count']} "
        f"singleton_track_count={summary['singleton_track_count']} "
        f"no_embedding_singleton_count={summary['no_embedding_singleton_count']} "
        f"held_incomplete_component_count={summary['held_incomplete_component_count']} "
        f"global_candidate_count={summary['global_candidate_count']} "
        "automatic_linking_enabled=false "
        "similarity_threshold=null "
        f"elapsed_sec={summary['elapsed_sec']:.3f} "
        f"output_dir={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
