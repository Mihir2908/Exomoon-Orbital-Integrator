#!/usr/bin/env bash
# launch_ec2_gpu.sh — Launch a g4dn.xlarge EC2 spot instance running the HNN GPU service.
#
# Creates ONLY new resources — does NOT modify existing ECS, Fargate, Step Functions,
# NLB, App Runner, or any existing security groups / ECR repos.
#
# New resources created:
#   - Security group: exomoon-hnn-gpu-sg  (port 8001 inbound, port 22 for SSH)
#   - EC2 spot instance: g4dn.xlarge (1x NVIDIA T4, 16GB RAM)
#
# Prerequisites:
#   1. Run ecr_push_gpu.sh first to build + push the image.
#   2. Set HNN_IMAGE below to the full ECR image URI printed by ecr_push_gpu.sh.
#   3. Set KEY_NAME to an existing EC2 key pair name (for SSH access if needed).
#
# Usage: bash launch_ec2_gpu.sh

set -euo pipefail

AWS_REGION="eu-west-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# ── CONFIGURE THESE ────────────────────────────────────────────────────────────
HNN_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/exomoon-hnn-gpu:latest"
KEY_NAME=""          # your EC2 key pair name, or leave empty to skip SSH
YOUR_IP=""           # your IP for SSH access, e.g. "203.0.113.5/32"; leave empty to skip
# ───────────────────────────────────────────────────────────────────────────────

INSTANCE_TYPE="g4dn.xlarge"
SG_NAME="exomoon-hnn-gpu-sg"

# Latest Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04) in eu-west-2
# This AMI has NVIDIA drivers, Docker, and nvidia-container-toolkit pre-installed.
echo "=== [1/5] Fetching latest Deep Learning AMI ==="
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --region "${AWS_REGION}" \
    --filters \
        "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)
echo "AMI: ${AMI_ID}"

echo "=== [2/5] Creating security group: ${SG_NAME} ==="
SG_ID=$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "HNN GPU inference service — port 8001" \
    --region "${AWS_REGION}" \
    --query 'GroupId' --output text)
echo "Security group: ${SG_ID}"

# Port 8001: HNN inference service (open to all — restrict to your IP in production)
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 8001 --cidr 0.0.0.0/0 \
    --region "${AWS_REGION}"

# Port 22: SSH (only if YOUR_IP is set)
if [[ -n "${YOUR_IP}" ]]; then
    aws ec2 authorize-security-group-ingress \
        --group-id "${SG_ID}" \
        --protocol tcp --port 22 --cidr "${YOUR_IP}" \
        --region "${AWS_REGION}"
fi

echo "=== [3/5] Writing user-data script ==="
# The DLAMI already has Docker and nvidia-container-toolkit.
# User-data: log into ECR, pull image, run container on port 8001.
USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -ex

# Log into ECR
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Pull GPU image
docker pull ${HNN_IMAGE}

# Run with GPU access, restart always
docker run -d \
    --gpus all \
    --restart always \
    -p 8001:8001 \
    --name hnn-gpu \
    -e ML_DEVICE=cuda \
    -e HNN_MODEL_DIR=/app/src/models_hnn_hill_hinge4 \
    ${HNN_IMAGE}
USERDATA
)

USER_DATA_B64=$(echo "${USER_DATA}" | base64 -w 0)

echo "=== [4/5] Launching g4dn.xlarge spot instance ==="
LAUNCH_SPEC=$(cat <<EOF
{
  "ImageId": "${AMI_ID}",
  "InstanceType": "${INSTANCE_TYPE}",
  "SecurityGroupIds": ["${SG_ID}"],
  "UserData": "${USER_DATA_B64}",
  "IamInstanceProfile": {"Name": "ec2-ecr-access-role"},
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {"VolumeSize": 60, "VolumeType": "gp3", "DeleteOnTermination": true}
    }
  ]
  $([ -n "${KEY_NAME}" ] && echo ", \"KeyName\": \"${KEY_NAME}\"" || echo "")
}
EOF
)

SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --spot-price "0.20" \
    --instance-count 1 \
    --type "one-time" \
    --region "${AWS_REGION}" \
    --launch-specification "${LAUNCH_SPEC}" \
    --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
    --output text)
echo "Spot request: ${SPOT_REQUEST}"

echo "=== [5/5] Waiting for instance to start (up to 5 min) ==="
aws ec2 wait spot-instance-request-fulfilled \
    --spot-instance-request-ids "${SPOT_REQUEST}" \
    --region "${AWS_REGION}"

INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
    --spot-instance-request-ids "${SPOT_REQUEST}" \
    --region "${AWS_REGION}" \
    --query 'SpotInstanceRequests[0].InstanceId' --output text)

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --region "${AWS_REGION}" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo ""
echo "=========================================================="
echo "  HNN GPU instance launched successfully"
echo "  Instance ID : ${INSTANCE_ID}"
echo "  Public IP   : ${PUBLIC_IP}"
echo "  Service URL : http://${PUBLIC_IP}:8001"
echo "  Health check: curl http://${PUBLIC_IP}:8001/health"
echo ""
echo "  NOTE: Docker container starts ~2-3 min after instance boots."
echo "  Set NEXT_PUBLIC_HNN_URL=http://${PUBLIC_IP}:8001 in your"
echo "  Next.js frontend to route HNN inference to this instance."
echo "=========================================================="

echo ""
echo "  NOTE: The EC2 instance role 'ec2-ecr-access-role' must exist"
echo "  with AmazonEC2ContainerRegistryReadOnly + AmazonS3ReadOnlyAccess."
echo "  Create it once if it doesn't exist:"
echo "    aws iam create-role --role-name ec2-ecr-access-role \\"
echo "      --assume-role-policy-document '{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"ec2.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'"
echo "    aws iam attach-role-policy --role-name ec2-ecr-access-role \\"
echo "      --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
echo "    aws iam create-instance-profile --instance-profile-name ec2-ecr-access-role"
echo "    aws iam add-role-to-instance-profile --instance-profile-name ec2-ecr-access-role \\"
echo "      --role-name ec2-ecr-access-role"
