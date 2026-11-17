#!/usr/bin/env bash
# Run manually on the Linux RTX 4090 server from the project root.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/plan/p0/remote-run"
mkdir -p "$RUN"

if ! command -v sumo >/dev/null 2>&1; then
  echo "SUMO executable not found in PATH" > "$RUN/environment.log"
  printf '1\n' > "$RUN/environment.exitcode"
  exit 1
fi

(
  set -e
  date -Is
  uname -a
  cat /etc/os-release
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  command -v sumo
  sumo --version
  command -v python
  python --version
  python - <<'PY'
import torch
import traci, sumolib, gym
print('torch.__version__:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('torch.cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('torch.cuda.device_name:', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert 'RTX 4090' in torch.cuda.get_device_name(0), 'GPU 0 is not RTX 4090'
PY
  python -m pip freeze
) > "$RUN/environment.log" 2>&1
env_status=$?
printf '%s\n' "$env_status" > "$RUN/environment.exitcode"
if [ "$env_status" -ne 0 ]; then
  echo "Environment probe failed; see $RUN/environment.log"
  exit "$env_status"
fi

python - "$ROOT" <<'PY' > "$RUN/inputs.log" 2>&1
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
snapshot = json.loads((root / 'plan/p0/snapshot.json').read_text(encoding='utf-8'))
for item in snapshot['files']:
    for key in ('source', 'copy'):
        path = root / item[key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
print(f"Verified {len(snapshot['files'])} reference/work-copy file pairs")
PY
input_status=$?
printf '%s\n' "$input_status" > "$RUN/inputs.exitcode"
if [ "$input_status" -ne 0 ]; then
  echo "Input verification failed; see $RUN/inputs.log"
  exit "$input_status"
fi

if [ -e "$RUN/work" ]; then
  echo "Refusing to overwrite $RUN/work"
  exit 1
fi
mkdir "$RUN/work"
cp -a "$ROOT/src/sac_rwb/." "$RUN/work/"
cd "$RUN/work" || exit 1
printf '%s\n' 'timeout 20m python train_sac.py --epochs 1 --max_e_steps 100 --save_interval 10000 --dvc cuda --seed 0' > "$RUN/command.txt"
timeout 20m python train_sac.py --epochs 1 --max_e_steps 100 --save_interval 10000 --dvc cuda --seed 0 > "$RUN/train.log" 2>&1
train_status=$?
printf '%s\n' "$train_status" > "$RUN/train.exitcode"
echo "Training smoke exit code: $train_status; logs: $RUN"
exit "$train_status"
