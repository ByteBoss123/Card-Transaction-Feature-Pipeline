"""
Step 4: Export a stratified sample to Parquet for pandas/XGBoost training.

Given the extreme class imbalance (~0.12% fraud) and typical local-machine RAM limits,
this takes ALL fraud transactions plus a ~15:1 random sample of non-fraud transactions,
rather than loading the full 24M rows into pandas. This is a disclosed, deliberate
compute-constraint decision — not an attempt to represent full-population training.
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

OUTPUT_DIR = os.environ.get("TXN_OUTPUT_DIR", "./output/parquet")
NONFRAUD_RATIO = float(os.environ.get("TXN_NONFRAUD_RATIO", "15"))

spark = SparkSession.builder.appName("CardTxn-ExportSample").config(
    "spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g")
).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

feat = spark.read.parquet(f"{OUTPUT_DIR}/features")

n_total = feat.count()
n_fraud = feat.filter(F.col("Is_Fraud") == 1).count()
print(f"Total: {n_total}, Fraud: {n_fraud}, Fraud rate: {n_fraud/n_total:.6f}")

target_nonfraud = n_fraud * NONFRAUD_RATIO
nonfraud_frac = min(1.0, target_nonfraud / (n_total - n_fraud))
print(f"Sampling non-fraud at fraction: {nonfraud_frac:.6f}")

fraud_rows = feat.filter(F.col("Is_Fraud") == 1)
nonfraud_rows = feat.filter(F.col("Is_Fraud") == 0).sample(withReplacement=False, fraction=nonfraud_frac, seed=42)

sample = fraud_rows.unionByName(nonfraud_rows)

model_cols = [
    "Is_Fraud", "Amount", "Use_Chip", "MCC", "Has_Error",
    "seconds_since_last_txn", "txn_count_trailing5", "txn_count_trailing20",
    "avg_amount_trailing5", "avg_amount_trailing20", "stddev_amount_trailing20",
    "distinct_mcc_trailing20", "amount_zscore_vs_trailing20", "card_txn_seq_num",
    "merchant_txn_count", "merchant_fraud_rate", "Merchant_State", "Zip",
]

sample_out = sample.select(*model_cols)
sample_out.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/model_sample")

n_sample = sample_out.count()
n_sample_fraud = sample_out.filter(F.col("Is_Fraud") == 1).count()
print(f"Sample rows: {n_sample}, Sample fraud rows: {n_sample_fraud}")
