#!/bin/bash
set -euo pipefail

BUCKET="${TXN_BUCKET:?Set TXN_BUCKET, same value as step 1}"
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${EC2_KEY_NAME:?Set EC2_KEY_NAME to an existing EC2 key pair name (for SSH access if needed)}"
SUBNET_ID="${EMR_SUBNET_ID:?Set EMR_SUBNET_ID to a subnet in your default VPC (aws ec2 describe-subnets)}"

S3_SRC="s3://${BUCKET}/card-txn/src"
S3_DATA="s3://${BUCKET}/card-txn/data"
S3_OUTPUT="s3://${BUCKET}/card-txn/output"
S3_LOGS="s3://${BUCKET}/card-txn/logs"

ENV_ARGS='"--conf","spark.yarn.appMasterEnv.TXN_DATA_DIR='"${S3_DATA}"'","--conf","spark.executorEnv.TXN_DATA_DIR='"${S3_DATA}"'","--conf","spark.yarn.appMasterEnv.TXN_OUTPUT_DIR='"${S3_OUTPUT}"'/parquet","--conf","spark.executorEnv.TXN_OUTPUT_DIR='"${S3_OUTPUT}"'/parquet","--conf","spark.yarn.appMasterEnv.TXN_WAREHOUSE_DIR='"${S3_OUTPUT}"'/hive_warehouse","--conf","spark.executorEnv.TXN_WAREHOUSE_DIR='"${S3_OUTPUT}"'/hive_warehouse","--conf","spark.yarn.appMasterEnv.TXN_USE_GLUE_CATALOG=true","--conf","spark.executorEnv.TXN_USE_GLUE_CATALOG=true"'

cat > ./emr_steps.json <<EOF
[
  {
    "Name": "01-Ingest",
    "ActionOnFailure": "TERMINATE_CLUSTER",
    "Jar": "command-runner.jar",
    "Args": ["spark-submit", "--deploy-mode", "cluster", ${ENV_ARGS}, "${S3_SRC}/01_ingest.py"]
  },
  {
    "Name": "02-Features",
    "ActionOnFailure": "TERMINATE_CLUSTER",
    "Jar": "command-runner.jar",
    "Args": ["spark-submit", "--deploy-mode", "cluster", ${ENV_ARGS}, "${S3_SRC}/02_features.py"]
  },
  {
    "Name": "03-Validate",
    "ActionOnFailure": "CONTINUE",
    "Jar": "command-runner.jar",
    "Args": ["spark-submit", "--deploy-mode", "cluster", ${ENV_ARGS}, "${S3_SRC}/03_validate.py"]
  },
  {
    "Name": "04-ExportSample",
    "ActionOnFailure": "CONTINUE",
    "Jar": "command-runner.jar",
    "Args": ["spark-submit", "--deploy-mode", "cluster", ${ENV_ARGS}, "${S3_SRC}/04_export_sample.py"]
  },
  {
    "Name": "06-HiveLayer",
    "ActionOnFailure": "CONTINUE",
    "Jar": "command-runner.jar",
    "Args": ["spark-submit", "--deploy-mode", "cluster", ${ENV_ARGS}, "${S3_SRC}/06_hive_layer.py"]
  }
]
EOF

echo "Generated ./emr_steps.json:"
cat ./emr_steps.json
echo ""

aws emr create-cluster \
  --name "Card-Transaction-Feature-Pipeline" \
  --release-label emr-7.7.0 \
  --applications Name=Spark Name=Hive \
  --ec2-attributes KeyName="${KEY_NAME}",SubnetId="${SUBNET_ID}" \
  --instance-groups \
      InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
      InstanceGroupType=CORE,InstanceCount=2,InstanceType=m5.xlarge \
  --use-default-roles \
  --log-uri "${S3_LOGS}" \
  --region "${REGION}" \
  --auto-terminate \
  --steps file://./emr_steps.json

echo ""
echo "Cluster launching. Track progress with:"
echo "  aws emr list-clusters --active --region ${REGION}"
