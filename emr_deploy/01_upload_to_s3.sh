#!/bin/bash
# Step 1: Upload the dataset and pipeline scripts to S3.
# Run this from your own machine with AWS CLI configured (aws configure).
set -euo pipefail

BUCKET="${TXN_BUCKET:?Set TXN_BUCKET, e.g. export TXN_BUCKET=my-bucket}"
REGION="${AWS_REGION:-us-east-1}"

echo "Creating bucket s3://${BUCKET} in ${REGION} (skips if it already exists)..."
aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}" \
  $( [ "${REGION}" != "us-east-1" ] && echo "--create-bucket-configuration LocationConstraint=${REGION}" ) \
  2>/dev/null || echo "Bucket already exists or you own it — continuing."

echo "Uploading pipeline scripts..."
aws s3 cp ./src/ "s3://${BUCKET}/card-txn/src/" --recursive --exclude "*.pyc"

echo "Uploading dataset (this is the 2.2GB CSV — takes a few minutes)..."
aws s3 cp ./data/card_transaction.v1.csv "s3://${BUCKET}/card-txn/data/card_transaction.v1.csv"

echo "Done. Data lives at s3://${BUCKET}/card-txn/data/"
echo "Scripts live at s3://${BUCKET}/card-txn/src/"
