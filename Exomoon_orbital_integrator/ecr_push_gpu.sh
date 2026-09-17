#!/usr/bin/env bash
# ecr_push_gpu.sh — Build gpu.Dockerfile and push to a NEW ECR repo (exomoon-hnn-gpu).
# Does NOT touch existing ECR repos (exomoon-agent, exomoon-batch).
#
# Usage: bash ecr_push_gpu.sh
# Prerequisites: aws cli configured, docker buildx available, eu-west-2 access.

set -euo pipefail

AWS_REGION="eu-west-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="exomoon-hnn-gpu"
IMAGE_TAG="latest"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"

echo "=== [1/4] Ensuring ECR repository exists: ${REPO_NAME} ==="
aws ecr describe-repositories --repository-names "${REPO_NAME}" \
    --region "${AWS_REGION}" > /dev/null 2>&1 || \
aws ecr create-repository \
    --repository-name "${REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --query 'repository.repositoryUri' --output text

echo "=== [2/4] Logging into ECR ==="
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${REGISTRY}"

echo "=== [3/4] Building GPU image (gpu.Dockerfile) ==="
# Run from Exomoon_orbital_integrator/ directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build \
    -f "${SCRIPT_DIR}/gpu.Dockerfile" \
    -t "${FULL_IMAGE}" \
    "${SCRIPT_DIR}"

echo "=== [4/4] Pushing to ECR: ${FULL_IMAGE} ==="
docker push "${FULL_IMAGE}"

echo ""
echo "Done. Image: ${FULL_IMAGE}"
echo "Use this value for HNN_IMAGE in launch_ec2_gpu.sh"
