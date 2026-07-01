#!/bin/sh
set -eu

pip install -r requirements.txt
mkdir -p logs outputs/traces

export PYTHONPYCACHEPREFIX=/tmp/semgate-pycache
PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_ROOT"
export SEMROUTE_API_KEY=dev-key
export GATEWAY_REQUEST_TIMEOUT_S=5
export GATEWAY_RATE_LIMIT_BURST_CAPACITY=20
export GATEWAY_LOG_PATH=logs/gateway.jsonl
export GATEWAY_WORKFLOW_PROFILES_PATH=configs/workflow_profiles.json
export GATEWAY_CONSUMERS_PATH=configs/consumers.json
export GATEWAY_TRACE_LOG_PATH=logs/trace_events.jsonl
export GATEWAY_TRACE_OUTPUT_DIR=outputs/traces
export GATEWAY_TOOL_AUDIT_LOG_PATH=logs/tool_audit.jsonl
export AGENT_ORCHESTRATOR_URL=http://127.0.0.1:8010/invoke
export RAG_SERVICE_URL=http://127.0.0.1:8020
export TOOL_SERVICE_URL=http://127.0.0.1:8030

python -m uvicorn services.rag_service.main:app --host 127.0.0.1 --port 8020 >/tmp/semgate-rag.log 2>&1 &
RAG_PID=$!
python -m uvicorn services.tool_service.main:app --host 127.0.0.1 --port 8030 >/tmp/semgate-tool.log 2>&1 &
TOOL_PID=$!
python -m uvicorn services.model_backend.main:app --host 127.0.0.1 --port 8040 >/tmp/semgate-model.log 2>&1 &
MODEL_PID=$!
python -m uvicorn services.agent_orchestrator.main:app --host 127.0.0.1 --port 8010 >/tmp/semgate-agent.log 2>&1 &
AGENT_PID=$!
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >/tmp/semgate-gateway.log 2>&1 &
GATEWAY_PID=$!

cleanup() {
  kill "$GATEWAY_PID" "$AGENT_PID" "$MODEL_PID" "$TOOL_PID" "$RAG_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 5
python scripts/acceptance_v10_phase1.py --all --request-delay-s 0.1
