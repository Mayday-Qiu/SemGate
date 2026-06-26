# SemRoute-Gateway

SemRoute-Gateway 是一个面向多模型、RAG 与 Agent 服务的轻量 AI Gateway 策略治理原型。当前处于 MVP-A-03：Resilience & Traffic Control 阶段。

项目基准环境：

- Python 3.9.21
- conda 环境名：`qiuqiupytorch`
- 本地开发优先使用 conda 环境；Linux/WSL2 或服务器部署时优先使用 Docker Compose。

在已有 conda 环境中安装依赖：

```bash
conda activate qiuqiupytorch
python --version
pip install -r requirements.txt
```

如果需要用 `environment.yml` 对齐依赖：

```bash
conda env update -n qiuqiupytorch -f environment.yml
```

Docker 启动：

```bash
docker compose up --build
```

Dockerfile 使用 Python 3.9 系镜像，和本项目基准 Python 版本保持一致。

当前 Gateway 默认启用内存版入口限流、本地熔断器和最多一次 fallback。常用环境变量：

```bash
GATEWAY_RATE_LIMIT_ENABLED=true
GATEWAY_RATE_LIMIT_REPLENISH_RATE=1
GATEWAY_RATE_LIMIT_BURST_CAPACITY=5
GATEWAY_FALLBACK_MAX_ATTEMPTS=2
GATEWAY_CIRCUIT_BREAKER_ENABLED=true
```

生成静态 service profile：

```bash
python scripts/generate_profiles.py --output outputs/profiles/service_profiles.json
```

切换路由策略：

```bash
GATEWAY_POLICY=fixed docker compose up --build
GATEWAY_POLICY=round_robin docker compose up --build
GATEWAY_POLICY=profile_aware docker compose up --build
```

示例请求：

```bash
curl -X POST http://localhost:8000/invoke ^
  -H "content-type: application/json" ^
  -H "x-api-key: dev-key" ^
  -d "{\"user_id\":\"u001\",\"tenant_id\":\"tenant_basic\",\"task_type\":\"summary\",\"input\":\"hello semroute\"}"
```

当前已加入 fixed、round-robin、profile-aware 三种路由策略、静态 service profile 生成、入口 rate limiting、timeout/fallback 和轻量 circuit breaker。后续 MVP-A 会继续补充 benchmark 汇总和更完整的运行指标分析。

Resilience 检查脚本：

```bash
python scripts/resilience_probe.py --file data/resilience_requests.jsonl --repeat 3
```
