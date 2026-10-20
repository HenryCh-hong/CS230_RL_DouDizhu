#!/usr/bin/env bash
# Stage 4 on AWS, step 1 of 3, run ON THE MAC: launch one on-demand c7i.4xlarge and ship the code.
# NOT APPROVED YET. It refuses to run without --approved, which is passed only after approving the
# proposal in docs/STAGE4_AWS_PROPOSAL.md. Needs the AWS CLI configured (aws configure) on the Mac.
#
#   cloud/stage4/launch.sh --approved
#
# The instance terminates itself after HARD_CAP_MIN minutes whatever happens (shutdown -h plus
# instance-initiated-shutdown-behavior terminate), and its disk is deleted with it.
set -euo pipefail
[[ "${1:-}" == "--approved" ]] || { echo "Refusing: Stage 4 AWS compute is not approved (pass --approved)."; exit 2; }

REGION=${REGION:-us-west-2}
TYPE=${TYPE:-c7i.4xlarge}            # 16 vCPU, 32 GiB, $0.714/h on demand (AWS price list, checked 2026-10-19)
HARD_CAP_MIN=${HARD_CAP_MIN:-150}    # the instance powers off, and so terminates, after this many minutes
KEY=${KEY:-cs230-stage4}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/cloud/stage4/.state"; mkdir -p "$OUT"

[[ -z "$(git -C "$ROOT" status --porcelain -- DouZero)" ]] || {
  echo "DouZero/ has uncommitted or untracked files: commit first, so the run pins its exact code."; exit 3; }
COMMIT=$(git -C "$ROOT" rev-parse HEAD)

# Key pair and a security group that admits SSH from this Mac's address only.
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY" >/dev/null 2>&1; then
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY" --query KeyMaterial --output text > ~/.ssh/$KEY.pem
  chmod 600 ~/.ssh/$KEY.pem
fi
MYIP=$(curl -fsS https://checkip.amazonaws.com)/32
SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=cs230-stage4 \
       --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [[ -z "$SG" || "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name cs230-stage4 \
         --description "CS230 Stage 4: SSH from one IP only" --query GroupId --output text)
fi
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 22 \
  --cidr "$MYIP" 2>/dev/null || true

AMI=$(aws ssm get-parameters --region "$REGION" \
        --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
        --query 'Parameters[0].Value' --output text)
# User data: the hard cap is armed before anything else runs.
USERDATA=$(printf '#!/bin/bash\nshutdown -h +%s "CS230 Stage 4 hard cap"\n' "$HARD_CAP_MIN")
ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
       --security-group-ids "$SG" --instance-initiated-shutdown-behavior terminate \
       --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
       --user-data "$USERDATA" \
       --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cs230-stage4},{Key=commit,Value=$COMMIT}]" \
       --query 'Instances[0].InstanceId' --output text)
echo "$ID" > "$OUT/instance_id"; echo "$REGION" > "$OUT/region"
# From here on, any failure terminates the instance at once instead of leaving it idle until the cap.
trap 'echo "Launch failed: terminating $ID"; aws ec2 terminate-instances --region "$REGION" --instance-ids "$ID" >/dev/null' ERR
echo "Launched $ID ($TYPE, $REGION); hard cap $HARD_CAP_MIN min. Waiting for it to boot..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
HOST=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
         --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "$HOST" > "$OUT/host"
SSH="ssh -i $HOME/.ssh/$KEY.pem -o StrictHostKeyChecking=accept-new ubuntu@$HOST"
for i in $(seq 30); do $SSH true 2>/dev/null && break; sleep 10; done

# Ship the committed code (a git bundle) and the DouZero-ADP checkpoints, about 19 MB.
git -C "$ROOT" bundle create "$OUT/code.bundle" HEAD
scp -i ~/.ssh/$KEY.pem "$OUT/code.bundle" "$ROOT/cloud/stage4/setup_instance.sh" "$ROOT/cloud/stage4/run_stage4.sh" \
    ubuntu@"$HOST":~/
tar czf "$OUT/adp.tgz" -C "$ROOT/DouZero/baselines" douzero_ADP
scp -i ~/.ssh/$KEY.pem "$OUT/adp.tgz" ubuntu@"$HOST":~/
$SSH "bash ~/setup_instance.sh $COMMIT"   # synchronous: a failed setup terminates the instance (trap above)
$SSH "nohup bash ~/run_stage4.sh > ~/stage4.log 2>&1 < /dev/null &"
trap - ERR
echo "Setup passed and the job started. Follow it with:  $SSH tail -f stage4.log"
echo "Pull progress with cloud/stage4/fetch_and_cleanup.sh --watch; finish with cloud/stage4/fetch_and_cleanup.sh"
