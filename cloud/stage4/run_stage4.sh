#!/usr/bin/env bash
# Stage 4 on AWS, ON THE INSTANCE: unit tests, the dataset, then the K=1 and K=2 predictors in parallel.
# The step timeouts add up to 100 minutes, inside the instance's 150-minute hard cap (shutdown -h), so
# the results are always packed before the cap. Pre-specified settings (docs/STAGE4_AWS_PROPOSAL.md):
# 33,000 games = 26,400 train / 3,300 val / 3,300 test, about 1e6 decisions; batch 256, Adam lr 1e-3,
# eval every 500 steps on the whole validation split, patience 5, at most 20,000 steps per seat, seed 0.
set -uo pipefail
cd ~/stage4/repo/DouZero
PY=../.venv/bin/python
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "unit tests (the same suite as on the Mac)"
timeout 10m $PY -m pytest tests -p no:cacheprovider -q -x || { log "unit tests failed: stopping"; exit 1; }
log "dataset: 33,000 games on 14 workers"
timeout 30m $PY generate_predictor_data.py --out ~/stage4/data --train_games 26400 --val_games 3300 --test_games 3300 \
  --workers 14 --shard_games 500 --eval_seeds 0 1 2 --eval_deals 10000 || { log "dataset failed: stopping"; exit 1; }
log "predictors: K=1 (M1 and M2c) and K=2 (M2) in parallel, 7 threads each"
COMMON="--data $HOME/stage4/data --savedir $HOME/stage4/runs --batch_size 256 --lr 1e-3 --eval_every 500 --patience 5 --max_steps 20000 --seed 0 --threads 7"
OMP_NUM_THREADS=7 timeout 60m $PY train_predictor.py --num_targets 1 --run_id stage4_k1 $COMMON > ~/stage4/k1.log 2>&1 &
K1=$!
OMP_NUM_THREADS=7 timeout 60m $PY train_predictor.py --num_targets 2 --run_id stage4_k2 $COMMON > ~/stage4/k2.log 2>&1 &
K2=$!
wait $K1; S1=$?
wait $K2; S2=$?
log "K=1 exit $S1, K=2 exit $S2; packing whatever exists (the dataset stays here; only its manifest comes back)"
tar czf ~/stage4/results.tgz -C ~/stage4 runs data/manifest.json k1.log k2.log 2>/dev/null || \
  tar czf ~/stage4/results.tgz -C ~/stage4 k1.log k2.log
log "done: $(du -h ~/stage4/results.tgz | cut -f1)"
[[ $S1 -eq 0 && $S2 -eq 0 ]]
