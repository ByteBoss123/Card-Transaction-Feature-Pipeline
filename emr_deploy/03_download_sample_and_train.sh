#!/bin/bash
# Step 3: After the EMR steps finish, download the stratified sample export
# (produced by 04_export_sample.py on the cluster) and train XGBoost locally.
# This matches the sandbox workflow exactly: Spark does the heavy lifting at
# 24.4M-row scale, XGBoost trains on the much smaller stratified sample —
# no reason to pay for cluster time during model training itself.
set -euo pipefail

BUCKET="${TXN_BUCKET:?Set TXN_BUCKET}"

mkdir -p ./output/parquet
aws s3 cp "s3://${BUCKET}/card-txn/output/parquet/sample" ./output/parquet/sample --recursive

echo "Sample downloaded. Running 05_train_model.py locally..."
export TXN_OUTPUT_DIR=./output/parquet
python3 src/05_train_model.py
