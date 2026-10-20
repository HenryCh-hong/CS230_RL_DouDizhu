# Stage 4 on AWS: proposal (NOT approved yet)

Nothing here has been run. No AWS resource exists. `cloud/stage4/launch.sh` refuses to start unless it is
given `--approved`. Prices are from AWS's own price-list files, checked and
cross-checked on 2026-10-19; spot discounts and uv's Python 3.14 build were *not* verifiable without an account.

## The job, measured on the Mac (one thread)

| step | measured on the Mac | at the planned size |
|---|---|---|
| DouZero-ADP self-play | 30.4 decisions per game, 0.6 ms per decision | 33,000 games → 1.0e6 decisions ≈ 10 CPU-minutes |
| raw `.npy` storage | 1,264 B per sample (landlord 1,186, peasants 1,297) | ≈ 1.3 GB (≈ 45 MB gzipped) |
| K=1 training step, batch 256 | 10.5 ms (2.9M parameters) | ≈ 1,050 steps per epoch per seat |
| K=2 training step, batch 256 | 15.3 ms (4.6M parameters) | as above |
| one batch in memory | 1.3 MB | — |

The workload is small, but the Mac is for correctness checks only, so it runs remotely.

## A–P

- **A. Instance.** One on-demand **c7i.4xlarge** (Intel Sapphire Rapids) in **us-west-2**: $0.714/h, the
  same price in us-east-1.
- **B. CPU, not GPU.** Data generation is Python game logic plus small forward passes, which only CPU
  cores speed up. The six predictors (3 seats × K=1, K=2) are small (2.9–4.7M parameters, batch 256), and
  a GPU would mostly wait on the data loader. A g5.xlarge GPU instance costs $1.006/h for 4 vCPUs,
  which is worse on both counts.
- **C. vCPU.** 16: 14 generation workers, then 2 training processes × 7 threads.
- **D. RAM.** 32 GiB. Each training process holds one seat's splits (≤ 0.5 GB) at a time, about 4 GB
  at the peak.
- **E. Disk.** 30 GB gp3 root volume (OS, Python and torch about 10 GB, data 1.3 GB, runs under 0.2 GB),
  deleted when the instance terminates.
- **F. Sample count.** 33,000 games split by game into 26,400 train, 3,300 validation and 3,300
  predictor-test games. That gives ≈ 0.80M / 0.10M / 0.10M decisions, about 267k / 33k / 33k per seat.
  Each game has its own seed in the Stage 4 namespace. After generation the script checks that no deal
  repeats and that none is among the 30,000 RL evaluation deals of `tools/dup_eval.py` seeds 0–2.
- **G. Storage size.** ≈ 1.3 GB raw `.npy` on the instance. It stays there: only `manifest.json` (the
  seeds, every deal hash and file sha256) comes back.
- **H. Generation runtime.** ≈ 3 min expected (1.0e6 × ≈1.2 ms per AWS core ÷ 14 workers), ≤ 10 min
  worst case; the step timeout is 40 min.
- **I. K=1 training runtime.** Early stopping should end each seat after 3–10 epochs. That is ≈ 5 min
  expected for all three seats, ≤ 15 min if every seat runs to 20,000 steps; the step timeout is 75 min.
- **J. K=2 training runtime.** In parallel with K=1: ≈ 8 min expected, ≤ 20 min worst case; the step
  timeout is 75 min.
- **K. Total cost.**
  - Wall time: setup ≈ 8 min, unit tests 1 min, data 3 min, training 8 min, packing 1 min, so about
    25 min expected and ≤ 45 min worst case.
  - Expected cost about **$0.30–0.55**: c7i.4xlarge at $0.714/h, plus disk at $0.08 per GB-month
    (≈ $0.01 for 2.5 h), plus the public IPv4 address at about $0.005/h (≈ $0.01), plus about 100 MB
    of download, inside the free 100 GB per month.
  - **Hard maximum ≈ $1.82**: the 150-minute self-terminate cap. Confirm the us-west-2 hourly rate in
    the EC2 console before approving.
- **L. Exact commands.**
  1. Commit everything (the launch script refuses a dirty `DouZero/`).
  2. On the Mac: `cloud/stage4/launch.sh --approved`. It:
     - creates a key pair and an SSH-only security group open to this Mac's IP;
     - launches Ubuntu 24.04 with the hard cap;
     - ships a git bundle of HEAD and the DouZero-ADP checkpoints, about 19 MB;
     - runs `cloud/stage4/setup_instance.sh`, which installs CPython 3.14.0, torch 2.14.1+cpu and
       numpy 2.5.4, and fails if any version differs from the Mac;
     - starts `cloud/stage4/run_stage4.sh`, which runs the unit suite, then
       `generate_predictor_data.py --train_games 26400 --val_games 3300 --test_games 3300 --workers 14`,
       then `train_predictor.py --num_targets 1` and `--num_targets 2` in parallel with the pre-specified
       settings (batch 256, Adam lr 1e-3, eval every 500 steps, patience 5, max 20,000 steps, seed 0).
  3. On the Mac: `cloud/stage4/fetch_and_cleanup.sh --watch`.
- **M. Timeout and auto-shutdown.**
  - The user data arms `shutdown -h +150` at boot, and the instance is launched with
    `--instance-initiated-shutdown-behavior terminate`, so after 150 minutes it powers off, which
    terminates it and deletes its disk, whatever state the job is in.
  - Every step has its own `timeout`: tests 10, data 30, training 60 minutes (K=1 and K=2 in
    parallel). They add up to 100 minutes, inside the cap, so the results are always packed first.
  - `launch.sh` refuses uncommitted or untracked files in `DouZero/`. It runs the environment setup
    synchronously, and **terminates the instance at once if any launch step fails**, so a failed setup
    never leaves an idle, billed machine.
  - Recommended once: an AWS Budgets email alert at $5. Its alert details are to be confirmed in the
    console.
- **N. How logs and checkpoints survive an interruption.**
  - Everything is written incrementally: dataset shards of 500 games, one `metrics.csv` row per
    evaluation, and each better predictor checkpoint saved atomically the moment it is found.
  - `fetch_and_cleanup.sh --watch` copies `runs/` and the log to the Mac every 5 minutes over rsync,
    so the latest checkpoints and metrics are local even if the instance dies. It only pulls: it never
    terminates anything, and it retries when SSH drops. This needs the Mac awake during the ~25-minute
    job. `fetch_and_cleanup.sh` without `--watch` fetches `results.tgz`, terminates the instance and
    lists anything still billed.
- **O. Artifacts that come back.** `results.tgz`, about 90 MB:
  - `runs/stage4_k1/` and `runs/stage4_k2/`: the selected `checkpoints/<seat>.ckpt` (about 12 MB and 19 MB
    per seat; 35 MB and 56 MB for the three seats together),
    `config.json`, `metrics.csv`, `baselines.json`, `test_metrics.json`, `summary.md` and `plots/`;
  - the dataset's `manifest.json`;
  - the logs.

  On the Mac they go to `DouZero/artifacts/aws_stage4/`. There we check each checkpoint's sha256, load
  `checkpoints/` with `load_predictors('next', …)` and `('derived_both', …)`, and look at
  `plots/index.html` and the examples page.
- **P. Spot.** **Not recommended.** It would save at most about $0.30 at the unverified ~47%
  discount. An interruption (2-minute notice through the IMDSv2 `spot/instance-action` endpoint)
  would mean rerunning: generation refuses to overwrite a dataset, and predictor training has no
  resume. On-demand at under $1 is simpler and safe.

Afterwards, `fetch_and_cleanup.sh` terminates the instance and lists any instance tagged `cs230-stage4`
or unattached volume still left in the region; both lists should be empty.

## What "the same environment" means

`setup_instance.sh` installs the same CPython (3.14.0), torch (2.14.1, CPU build) and numpy (2.5.4) as the
Mac, and refuses to continue if a version differs. The AWS games are still not byte-identical to games
generated on the Mac:
- the CPU type and the maths libraries differ (Apple Silicon versus Intel);
- the network's floating-point results can therefore differ in the last bits;
- that occasionally changes an argmax, and so a game.

Reproducibility is per platform. Rerunning the same commands on the same instance type gives the same
data. Every file's sha256 is in the manifest.
