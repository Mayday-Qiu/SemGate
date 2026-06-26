from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeatedly call SemRoute-Gateway for resilience checks.")
    parser.add_argument("--file", default="data/resilience_requests.jsonl", help="Path to JSONL request file.")
    parser.add_argument("--url", default="http://localhost:8000/invoke", help="Gateway invoke endpoint.")
    parser.add_argument("--api-key", default="dev-key", help="Value for x-api-key header.")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to replay the JSONL file.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests in seconds.")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} must be a JSON object.")
            yield payload


def main() -> None:
    args = parse_args()
    request_path = Path(args.file)
    payloads: List[Dict[str, Any]] = list(read_jsonl(request_path))
    headers = {
        "content-type": "application/json",
        "x-api-key": args.api_key,
    }

    with httpx.Client(timeout=args.timeout) as client:
        request_index = 0
        for round_index in range(1, args.repeat + 1):
            for payload in payloads:
                request_index += 1
                print(
                    f"\n=== request {request_index}: "
                    f"round={round_index} task_type={payload.get('task_type')} ==="
                )
                try:
                    response = client.post(args.url, headers=headers, json=payload)
                except httpx.HTTPError as exc:
                    print(f"request_error: {exc}")
                    continue

                print(f"http_status: {response.status_code}")
                try:
                    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print(response.text)

                if args.delay > 0:
                    time.sleep(args.delay)


if __name__ == "__main__":
    main()
