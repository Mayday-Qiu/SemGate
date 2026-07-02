# SemGateway

SemGateway 是面向 agentic / generative 任务的 Gateway Harness。

```text
knowledge_qa_workflow        快速知识问答，调用 RAG、证据检查和 citation
large_knowledge_qa_workflow  深度知识问答，子问题拆解、并行检索、证据聚合
coding_workflow              受控代码生成/解释/补丁建议，不直接写文件
media_generation_workflow    图像/视频生成，包含 prompt safety 和 asset metadata
document_writing_workflow    文档写作，通过 A2A 调 knowledge/media，禁止直接访问 RAG
```

## 核心能力

- `TaskContract`：把一次用户请求变成可执行、可验收的任务合同。
- `VerificationGate`：检查 trace、工具调用、citations、schema 和输出状态，避免 workflow 直接把未验收结果返回给用户。
- `TaskPlanner`：可选接入 Qwen Planner，复杂任务输出结构化 `TaskPlan`，简单任务走规则链路。
- `Tool Guard`：工具调用统一经过 schema、权限、timeout 和 audit。
- `Planner Knowledge Base`：离线使用 DeepEval、trace 和 LLM judge 结果沉淀规划知识。当前 Phase5 已完成 eval -> 归因 -> candidate experience 的过渡链路，后续 Phase6 会把它收束为统一的 `planning_knowledge`、`routing_knowledge` 和 `semantic_knowledge`。

## 快速运行

Mac 本机开发默认使用 Docker Desktop / Docker Compose 跑多服务。`.venv` 只用于本地脚本和离线评测。

```bash
cd "/Users/qiuqiu/Desktop/研三/Gateway/SemGateway"
GATEWAY_RATE_LIMIT_ENABLED=false GATEWAY_REQUEST_TIMEOUT_S=10 docker compose up --build
```

默认服务：

```text
gateway              :8000
agent_orchestrator   :8010
rag_service          :8020
tool_service         :8030
model_backend        :8040
```

健康检查：

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

后台运行：

```bash
GATEWAY_RATE_LIMIT_ENABLED=false GATEWAY_REQUEST_TIMEOUT_S=10 docker compose up --build -d
docker compose down
```

## 开启 Planner

如果要让 Gateway 调用LLM，

```env
TASK_PLANNER_ENABLED=true
SILICONFLOW_BASE_URL=
SILICONFLOW_API_KEY=replace_with_local_key
PLANNER_MODEL_ID=Qwen/Qwen3-8B
PLANNER_TEMPERATURE=0
PLANNER_TOP_P=0.8
PLANNER_MAX_TOKENS=1200
PLANNER_TIMEOUT_SECONDS=45
PLANNER_REPAIR_MAX_RETRIES=1
PLANNER_ENABLE_THINKING=false
```

## 本地脚本环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -U deepeval
python -c "import fastapi, pydantic, deepeval; print('env ok')"
```

`.venv/`、`logs/`、`outputs/` 都是本地生成物，默认不需要提交。

## 验收与评测

快速验收：

```bash
make demo
```

离线评测方案：

```bash
source .venv/bin/activate
python scripts/run_deepeval.py --cases data/task_eval_cases.jsonl --request-delay-s 0.5
```


```bash
python scripts/run_deepeval.py --mode preview --cases data/planner_eval_cases.jsonl --request-delay-s 0.5
```



```bash
python scripts/update_planner_memory.py --input outputs/eval/task_eval_results.jsonl
python scripts/compare_memory_ablation.py --baseline outputs/eval/no_memory.jsonl --candidate outputs/eval/with_memory.jsonl
```

主要输出：

```text
outputs/eval/task_eval_results.jsonl
outputs/reports/deepeval_report.md
outputs/reports/planner_memory_report.md        
outputs/reports/memory_ablation_report.md       
outputs/planner_memory/route_rules.json         
outputs/planner_memory/contract_rules.json      
```

## 代码地图

```text
app/main.py                              Gateway API、路由、合同、验收、trace 汇总
app/task_profile.py                      TaskProfileBuilder
app/planner_policy.py                    Planner 调用策略
app/planner_context.py                   PlannerContext 资源目录
app/task_planner.py                      Gateway 内直连 SiliconFlow 的 TaskPlanner
app/plan_validator.py                    TaskPlan schema/resource/A2A/permission 校验
app/agentic_router.py                    WorkflowProfile 路由选择
app/task_contract.py                     TaskContract 生成
app/verification.py                      VerificationGate
app/memory_planner.py                    Phase5 过渡知识应用器，后续收束为 Planner Knowledge
configs/workflow_profiles.json           五条 workflow profile
services/agent_orchestrator/workflows/   LangGraph workflow
services/rag_service/main.py             当前 RAG stub
services/tool_service/                   工具注册、权限、schema、audit
services/model_backend/main.py           mock / OpenAI-compatible model backend
eval/metrics/semgateway_metrics.py       DeepEval custom metrics
scripts/run_deepeval.py                  离线 eval runner
scripts/update_planner_memory.py         Phase5 candidate experience 生成脚本
```


## 问题



macOS 系统 Python 可能出现 `NotOpenSSLWarning`。优先使用 `.venv`；
