# SemGateway

Gateway Harness 原型。

```text
knowledge_qa_workflow        快速知识问答，调用 RAG、证据检查和 citation
large_knowledge_qa_workflow  深度知识问答，子问题拆解、并行检索、证据聚合
coding_workflow              受控代码生成/解释/补丁建议，不直接写文件
media_generation_workflow    图像/视频生成，包含 prompt safety 和 asset metadata
document_writing_workflow    文档写作，通过 A2A 调 knowledge/media，禁止直接访问 RAG
```

## 运行方式



启动服务：

```bash
cd "/Users/qiuqiu/Desktop/研三/Gateway/SemGateway"

GATEWAY_RATE_LIMIT_ENABLED=false GATEWAY_REQUEST_TIMEOUT_S=10 docker compose up --build
```

默认启动 5 个服务：

```text
gateway              :8000
agent_orchestrator   :8010
rag_service          :8020
tool_service         :8030
model_backend        :8040
```

另开终端检查健康状态：

```bash
curl http://localhost:8000/health
curl http://localhost:8010/health
curl http://localhost:8020/health
curl http://localhost:8030/health
curl http://localhost:8040/health
```

示例请求：

```bash
curl -X POST http://localhost:8000/v1/invoke \
  -H "content-type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{"user_id":"u001","tenant_id":"tenant_demo","task_type":"knowledge_qa","input":"According to project docs, explain why SemGateway needs routing, evidence, and citations.","metadata":{"knowledge_base":"project_docs","evidence_required":true}}'
```

后台启动和停止：

```bash
GATEWAY_RATE_LIMIT_ENABLED=false GATEWAY_REQUEST_TIMEOUT_S=10 docker compose up --build -d
docker compose down
```

## 本地 Python 环境

创建 `.venv` 用于运行脚本：

```bash
cd "/Users/qiuqiu/Desktop/研三/Gateway/SemGateway"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -U deepeval
```

验证：

```bash
python -c "import fastapi, pydantic, deepeval; print('env ok')"
```

`.venv/`、`logs/`、`outputs/` 都是本地生成物，默认不需要提交。

## 验收

Phase 1-3 快速验收：

```bash
make demo
```

`make demo` 调用 `scripts/demo_run.py --all`，会检查服务健康、Preview/Dry-run、五条 workflow、TaskContract、VerificationGate、Planner Memory、工具 schema/权限错误和 trace 文件。

Phase 4 DeepEval 离线评测：

```bash
source .venv/bin/activate

python scripts/run_deepeval.py --cases data/task_eval_cases.jsonl --request-delay-s 0.5
python scripts/update_planner_memory.py --input outputs/eval/task_eval_results.jsonl
```

DeepEval 只在离线 eval 中运行，不进入 `/v1/invoke` 在线主链路。在线请求只执行 Gateway 自己的 `VerificationGate`。DeepEval 输出用于生成：

```text
outputs/eval/task_eval_results.jsonl
outputs/reports/deepeval_report.md
outputs/reports/planner_memory_report.md
outputs/planner_memory/route_rules.json
outputs/planner_memory/contract_rules.json
```

## 核心文件

```text
app/main.py                              Gateway API、路由、合同、验收、trace 汇总
app/task_profile.py                      TaskProfileBuilder
app/agentic_router.py                    WorkflowProfile 路由选择
app/task_contract.py                     TaskContract 生成和 Planner Memory patch
app/verification.py                      VerificationGate
app/memory_planner.py                    Planner Memory 规则应用
configs/workflow_profiles.json           五条 workflow profile
services/agent_orchestrator/workflows/   LangGraph workflow
services/rag_service/main.py             当前 RAG stub
services/tool_service/                   工具注册、权限、schema、audit
services/model_backend/main.py           mock / OpenAI-compatible model backend
eval/metrics/semgateway_metrics.py       Phase 4 七个 DeepEval custom metrics
scripts/run_deepeval.py                  离线 eval runner
scripts/update_planner_memory.py         eval 结果到 Planner Memory 规则的适配
```

