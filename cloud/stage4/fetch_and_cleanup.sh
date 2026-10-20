#!/usr/bin/env bash
# Stage 4 on AWS, step 3 of 3, ON THE MAC.
#   cloud/stage4/fetch_and_cleanup.sh --watch   pull runs/ and the log every 5 min while the job runs; it never
#                                               terminates anything and keeps retrying when SSH drops
#   cloud/stage4/fetch_and_cleanup.sh           final pull of results.tgz, then terminate and check nothing is left
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STATE="$ROOT/cloud/stage4/.state"
ID=$(cat "$STATE/instance_id"); REGION=$(cat "$STATE/region"); HOST=$(cat "$STATE/host")
KEY=${KEY:-cs230-stage4}
DEST="$ROOT/DouZero/artifacts/aws_stage4"; mkdir -p "$DEST"
SSHOPT=(-i "$HOME/.ssh/$KEY.pem" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
state() { aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
            --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo unknown; }

if [[ "${1:-}" == "--watch" ]]; then
  while [[ "$(state)" == running ]]; do
    rsync -az -e "ssh ${SSHOPT[*]}" ubuntu@"$HOST":stage4/runs ubuntu@"$HOST":stage4.log "$DEST/" 2>/dev/null \
      || echo "$(date +%H:%M) pull failed (SSH?); retrying in 5 min"
    if ssh "${SSHOPT[@]}" ubuntu@"$HOST" test -f stage4/results.tgz 2>/dev/null; then
      echo "results.tgz is ready: run cloud/stage4/fetch_and_cleanup.sh to fetch it and terminate"; exit 0
    fi
    sleep 300
  done
  echo "The instance is $(state): partial copies are in $DEST"; exit 0
fi

if [[ "$(state)" == running ]]; then
  scp "${SSHOPT[@]}" ubuntu@"$HOST":stage4/results.tgz ubuntu@"$HOST":stage4.log "$DEST/" || \
    echo "WARNING: results.tgz not found; keeping the partial copies already pulled into $DEST"
  [[ -f "$DEST/results.tgz" ]] && tar xzf "$DEST/results.tgz" -C "$DEST"
fi
aws ec2 terminate-instances --region "$REGION" --instance-ids "$ID" >/dev/null
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$ID"
echo "Terminated $ID. Still running or billed in $REGION (both lists should be empty):"
aws ec2 describe-instances --region "$REGION" --filters Name=tag:Name,Values=cs230-stage4 \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId'
aws ec2 describe-volumes --region "$REGION" --filters Name=status,Values=available --query 'Volumes[].VolumeId'
echo "Results in $DEST. Verify each checkpoint's sha256 against its run's summary before using it."
