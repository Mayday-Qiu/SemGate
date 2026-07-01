from __future__ import annotations

import json
from typing import Any, Dict, List

from eval.metrics import (
    CitationSourceMetric,
    ContractComplianceMetric,
    RouteCorrectnessMetric,
    SchemaComplianceMetric,
    ToolComplianceMetric,
    TraceEventCoverageMetric,
    VerificationPassMetric,
)


def semgateway_metrics() -> List[Any]:
    return [
        RouteCorrectnessMetric(),
        ContractComplianceMetric(),
        TraceEventCoverageMetric(),
        ToolComplianceMetric(),
        VerificationPassMetric(),
        CitationSourceMetric(),
        SchemaComplianceMetric(),
    ]


def record_to_test_case(record: Dict[str, Any]) -> Any:
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=str(record.get("case_id", "")),
        actual_output=json.dumps(record, ensure_ascii=False),
        expected_output=str(record.get("case", {}).get("expected_workflow", "")),
    )
    setattr(test_case, "additional_metadata", {"record": record})
    return test_case
