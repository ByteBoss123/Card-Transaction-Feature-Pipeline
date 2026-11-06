"""
Step 3: Sanity-check the engineered features before modeling — row/card counts,
a manual spot-check of one card's rolling window values, null placement, and a
fraud-rate cross-check against the ingest step.
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

OUTPUT_DIR = os.environ.get("TXN_OUTPUT_DIR", "./output/parquet")

spark = SparkSession.builder.appName("CardTxn-Validate").config(
    "spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g")
).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

feat = spark.read.parquet(f"{OUTPUT_DIR}/features")

print("Total rows:", feat.count())
print("Distinct card_id:", feat.select("card_id").distinct().count())

print("\nSample: one card's transaction sequence with engineered features")
sample_card = feat.filter(F.col("card_txn_seq_num") <= 8).filter(F.col("card_id") == "0_0").orderBy("card_txn_seq_num")
sample_card.select(
    "card_txn_seq_num", "event_ts", "Amount", "seconds_since_last_txn",
    "txn_count_trailing5", "avg_amount_trailing5", "distinct_mcc_trailing20"
).show(10, truncate=False)

print("\nNull check on key engineered columns:")
feat.select(
    F.sum(F.col("seconds_since_last_txn").isNull().cast("int")).alias("null_seconds_since_last"),
    F.sum(F.col("txn_count_trailing5").isNull().cast("int")).alias("null_trailing5_count"),
    F.sum(F.col("merchant_fraud_rate").isNull().cast("int")).alias("null_merchant_fraud_rate"),
).show()

print("\nOverall fraud rate check (must match ingest step):")
feat.selectExpr("avg(Is_Fraud) as rate").show()
