from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eval JSONL results with MemoryPlanner off/on.")
    parser.add_argument("--baseline", required=True, help="Eval JSONL without the memory rules under test.")
    parser.add_argument("--candidate", required=True, help="Eval JSONL with the memory rules under test.")
    parser.add_argument("--report", default="outputs/reports/memory_ablation_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _by_case_id(_read_jsonl(Path(args.baseline)))
    candidate = _by_case_id(_read_jsonl(Path(args.candidate)))
    case_ids = sorted(set(baseline) & set(candidate))
    if not case_ids:
        raise SystemExit("no overlapping case_id values between baseline and candidate")

    metrics = sorted(
        {
            name
            for record in list(baseline.values()) + list(candidate.values())
            for name in (record.get("metric_scores") or {})
        }
    )
    lines = ["# Planner Memory Ablation Report", ""]
    lines.append(f"- compared_cases: {len(case_ids)}")
    lines.append(f"- baseline: {args.baseline}")
    lines.append(f"- candidate: {args.candidate}")
    lines.append("")
    lines.append("| metric | baseline_avg | candidate_avg | delta | baseline_pass | candidate_pass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for metric in metrics:
        base_scores = [_metric_score(baseline[case_id], metric) for case_id in case_ids]
        cand_scores = [_metric_score(candidate[case_id], metric) for case_id in case_ids]
        base_avg = _avg(base_scores)
        cand_avg = _avg(cand_scores)
        lines.append(
            f"| {metric} | {base_avg:.4f} | {cand_avg:.4f} | {cand_avg - base_avg:+.4f} | "
            f"{_pass_count(baseline, case_ids, metric)}/{len(case_ids)} | "
            f"{_pass_count(candidate, case_ids, metric)}/{len(case_ids)} |"
        )

    improved, regressed = _case_deltas(baseline, candidate, case_ids)
    lines.append("")
    lines.append(f"- improved_cases: {len(improved)}")
    lines.append(f"- regressed_cases: {len(regressed)}")
    if regressed:
        lines.append("")
        lines.append("## Regressed Cases")
        lines.append("")
        for case_id in regressed:
            lines.append(f"- {case_id}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote memory ablation report to {report_path}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records


def _by_case_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(record.get("case_id")): record for record in records if record.get("case_id")}


def _metric_score(record: Dict[str, Any], metric: str) -> float:
    try:
        return float((record.get("metric_scores") or {}).get(metric, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _pass_count(records: Dict[str, Dict[str, Any]], case_ids: List[str], metric: str) -> int:
    return sum(1 for case_id in case_ids if metric not in set(records[case_id].get("failed_metrics") or []))


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _case_deltas(
    baseline: Dict[str, Dict[str, Any]],
    candidate: Dict[str, Dict[str, Any]],
    case_ids: List[str],
) -> tuple[List[str], List[str]]:
    improved = []
    regressed = []
    for case_id in case_ids:
        base_failed = _failed_set(baseline[case_id])
        cand_failed = _failed_set(candidate[case_id])
        if len(cand_failed) < len(base_failed):
            improved.append(case_id)
        if len(cand_failed) > len(base_failed) or bool(cand_failed - base_failed):
            regressed.append(case_id)
    return improved, regressed


def _failed_set(record: Dict[str, Any]) -> Set[str]:
    return {str(item) for item in record.get("failed_metrics") or []}


if __name__ == "__main__":
    main()
