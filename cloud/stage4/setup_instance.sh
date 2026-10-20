#!/usr/bin/env bash
# Stage 4 on AWS, ON THE INSTANCE: the same Python and packages as the Mac, then the code at the
# given commit. Fails loudly if a version differs, before any compute is spent.
set -euo pipefail
COMMIT=$1
sudo apt-get update -qq && sudo apt-get install -y -qq git build-essential >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.14.0                      # the Mac runs CPython 3.14.0
mkdir -p ~/stage4 && cd ~/stage4
git clone -q ~/code.bundle repo && git -C repo checkout -q "$COMMIT"
uv venv --python 3.14.0 repo/.venv
uv pip install --python repo/.venv/bin/python -q --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
uv pip install --python repo/.venv/bin/python -q numpy==2.5.4 GitPython pytest==9.1.1
mkdir -p repo/DouZero/baselines && tar xzf ~/adp.tgz -C repo/DouZero/baselines
repo/.venv/bin/python - <<'EOF'
import sys, numpy, torch
assert sys.version.split()[0] == '3.14.0', sys.version
assert torch.__version__.startswith('2.14.1') and numpy.__version__ == '2.5.4', (torch.__version__, numpy.__version__)
print('environment ok:', sys.version.split()[0], torch.__version__, numpy.__version__, 'threads', torch.get_num_threads())
EOF
