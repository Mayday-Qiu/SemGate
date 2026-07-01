#!/bin/sh
set -eu

pip install -r requirements.txt
export PYTHONPYCACHEPREFIX=/tmp/semgate-pycache
python -m compileall app services scripts
python -c "import app.main; import services.agent_orchestrator.main; import services.tool_service.main; print('v1 phase1 imports ok')"
