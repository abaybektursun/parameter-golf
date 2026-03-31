#!/bin/bash
# Continuously tries to launch a p5.4xlarge (1x H100) across all us-east-1 AZs in parallel.
# Stops as soon as one succeeds. Sends ntfy notification on success.
# Usage: nohup ./agent/acquire_gpu.sh &

AMI="ami-00eda30d2e97acac9"
INSTANCE_TYPE="p5.4xlarge"
KEY_NAME="parameter-golf"
SG="sg-0ff9d83db750a5a32"
PROFILE="fuelos"
REGION="us-east-1"
NTFY_TOPIC="${NTFY_TOPIC:-parameter-golf-alerts}"
LOCK="/tmp/gpu_acquired.lock"

AZS=("us-east-1a" "us-east-1b" "us-east-1c" "us-east-1d" "us-east-1e" "us-east-1f")

rm -f "$LOCK"

try_az() {
  local az=$1
  result=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG" \
    --placement "AvailabilityZone=$az" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=parameter-golf-h100}]' \
    2>&1)

  if echo "$result" | grep -q "InstanceId"; then
    instance_id=$(echo "$result" | grep -o '"InstanceId": "[^"]*"' | head -1 | cut -d'"' -f4)
    echo "[$(date)] SUCCESS in $az: $instance_id"
    touch "$LOCK"
    # Get public IP (wait for it to be assigned)
    sleep 10
    ip=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" \
      --instance-ids "$instance_id" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
    echo "[$(date)] Instance $instance_id in $az, IP: $ip"
    curl -s -d "H100 ACQUIRED: $instance_id in $az, IP: $ip" "https://ntfy.sh/$NTFY_TOPIC"
    echo "$instance_id $az $ip" > /tmp/gpu_instance.txt
  fi
}

attempt=0
while [ ! -f "$LOCK" ]; do
  attempt=$((attempt + 1))
  echo "[$(date)] Attempt $attempt — trying all AZs in parallel..."

  for az in "${AZS[@]}"; do
    [ -f "$LOCK" ] && break
    try_az "$az" &
  done
  wait

  [ -f "$LOCK" ] && break
  echo "[$(date)] No capacity. Retrying in 30s..."
  sleep 30
done

echo "[$(date)] Done. Instance details in /tmp/gpu_instance.txt"
