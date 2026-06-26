from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


METRIC_FIELDS = {
    "avg_latency_ms",
    "p95_latency_ms",
    "success_rate",
    "citation_coverage",
    "tool_success_rate",
    "error_rate",
    "cost_level",
    "healthy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update seed workflow profiles from eval metrics.")
    parser.add_argument("--profiles", default="configs/workflow_profiles.json")
    parser.add_argument("--metrics", default="outputs/profiles/workflow_metrics.json")
    parser.add_argument("--output", default="configs/workflow_profiles.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles_path = Path(args.profiles)
    metrics_path = Path(args.metrics)
    output_path = Path(args.output)

    payload = json.loads(profiles_path.read_text(encoding="utf-8"))
    if not metrics_path.exists():
        print(f"No metrics file found at {metrics_path}; profiles left unchanged.")
        return

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_by_workflow: Dict[str, Dict[str, Any]] = {
        item["workflow_id"]: item for item in metrics_payload.get("workflows", [])
    }

    updated_count = 0
    for profile in payload.get("profiles", []):
        workflow_metrics = metrics_by_workflow.get(profile["workflow_id"])
        if workflow_metrics is None:
            continue
        for field in METRIC_FIELDS:
            if field in workflow_metrics:
                profile[field] = workflow_metrics[field]
        profile["metrics_source"] = "eval"
        profile["updated_by_eval_run_id"] = metrics_payload.get("eval_run_id")
        profile["metrics_report_path"] = metrics_payload.get("metrics_report_path")
        updated_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {updated_count} workflow profiles from {metrics_path}.")


if __name__ == "__main__":
    main()
