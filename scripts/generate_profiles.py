from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROFILE_ROWS: List[Dict[str, Any]] = [
    {
        "backend_id": "mock_fast",
        "task_type": "summary",
        "heuristic_quality_score": 0.72,
        "avg_latency_ms": 130,
        "p95_latency_ms": 180,
        "error_rate": 0.02,
        "timeout_rate": 0.00,
    },
    {
        "backend_id": "mock_quality",
        "task_type": "summary",
        "heuristic_quality_score": 0.86,
        "avg_latency_ms": 560,
        "p95_latency_ms": 750,
        "error_rate": 0.01,
        "timeout_rate": 0.00,
    },
    {
        "backend_id": "mock_unstable",
        "task_type": "summary",
        "heuristic_quality_score": 0.58,
        "avg_latency_ms": 650,
        "p95_latency_ms": 1100,
        "error_rate": 0.20,
        "timeout_rate": 0.12,
    },
    {
        "backend_id": "mock_fast",
        "task_type": "semantic_explain",
        "heuristic_quality_score": 0.68,
        "avg_latency_ms": 150,
        "p95_latency_ms": 200,
        "error_rate": 0.02,
        "timeout_rate": 0.00,
    },
    {
        "backend_id": "mock_quality",
        "task_type": "semantic_explain",
        "heuristic_quality_score": 0.90,
        "avg_latency_ms": 600,
        "p95_latency_ms": 800,
        "error_rate": 0.01,
        "timeout_rate": 0.00,
    },
    {
        "backend_id": "mock_unstable",
        "task_type": "semantic_explain",
        "heuristic_quality_score": 0.60,
        "avg_latency_ms": 700,
        "p95_latency_ms": 1200,
        "error_rate": 0.20,
        "timeout_rate": 0.12,
    },
    {
        "backend_id": "mock_quality",
        "task_type": "rag_qa",
        "heuristic_quality_score": 0.88,
        "avg_latency_ms": 620,
        "p95_latency_ms": 850,
        "error_rate": 0.01,
        "timeout_rate": 0.00,
    },
    {
        "backend_id": "mock_unstable",
        "task_type": "rag_qa",
        "heuristic_quality_score": 0.55,
        "avg_latency_ms": 700,
        "p95_latency_ms": 1200,
        "error_rate": 0.20,
        "timeout_rate": 0.12,
    },
    {
        "backend_id": "mock_unstable",
        "task_type": "tool_call",
        "heuristic_quality_score": 0.58,
        "avg_latency_ms": 650,
        "p95_latency_ms": 1100,
        "error_rate": 0.20,
        "timeout_rate": 0.12,
    },
    {
        "backend_id": "mock_quality",
        "task_type": "text_to_video",
        "heuristic_quality_score": 0.96,
        "avg_latency_ms": 900,
        "p95_latency_ms": 1200,
        "error_rate": 0.01,
        "timeout_rate": 0.00,
    },
    {
        "backend_id": "mock_unstable",
        "task_type": "text_to_video",
        "heuristic_quality_score": 0.45,
        "avg_latency_ms": 800,
        "p95_latency_ms": 1400,
        "error_rate": 0.20,
        "timeout_rate": 0.12,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static mock service profiles.")
    parser.add_argument(
        "--output",
        default="outputs/profiles/service_profiles.json",
        help="Path to generated service profile JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "source": "static_mock_profile_v1",
        "profiles": [
            dict(row, last_updated=generated_at)
            for row in PROFILE_ROWS
        ],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
