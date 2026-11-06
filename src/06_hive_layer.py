"""
Step 6 (added): Register the engineered features table in a real Hive
metastore (via Spark's enableHiveSupport(), Derby-backed embedded metastore)
and run genuine HiveQL analytical queries against it.

Why this exists: demonstrates Hive metastore / HiveQL usage directly, distinct
from the plain Spark DataFrame API used in 01-05. This is the same
Spark-on-Hive pattern used on EMR clusters that run Hive alongside Spark for
table/schema management — a managed metastore that both engines share.

Design choice, stated plainly: the features table is registered as an
EXTERNAL table pointing at the existing Parquet location (LOCATION clause),
not a managed table. A managed table would COPY the full 24.4M-row,
1.3GB Parquet dataset into the Hive warehouse directory, doubling disk
usage for no benefit. Pointing an external table at data that already
exists in place is the standard practice for large datasets in a real
Hive/EMR environment, not just a workaround for sandbox disk limits.
"""
import os, json, time
from pyspark.sql import SparkSession

OUTPUT_DIR = os.environ.get("TXN_OUTPUT_DIR", "./output/parquet")
WAREHOUSE_DIR = os.environ.get("TXN_WAREHOUSE_DIR", "./output/hive_warehouse")
RESULTS_DIR = "./results"

spark_builder = (
    SparkSession.builder
    .appName("CardTxn-HiveLayer")
    .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "3g"))
    .config("spark.sql.warehouse.dir", WAREHOUSE_DIR)
    .config("spark.sql.shuffle.partitions", "16")
)
# On EMR, use Glue Data Catalog as the Hive metastore (real, shared, persists
# past cluster termination) instead of the sandbox's embedded Derby metastore.
if os.environ.get("TXN_USE_GLUE_CATALOG", "false").lower() == "true":
    spark_builder = spark_builder.config(
        "spark.hadoop.hive.metastore.client.factory.class",
        "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory",
    )
spark = spark_builder.enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def to_abs_path(p):
    """S3 paths pass through unchanged; local paths get resolved to absolute
    (Hive's LOCATION clause needs an absolute path for local filesystem, but
    must NOT be rewritten for s3:// URIs)."""
    if p.startswith("s3://") or p.startswith("s3a://"):
        return p
    return os.path.abspath(p)

abs_features_path = to_abs_path(f"{OUTPUT_DIR}/features")

spark.sql("CREATE DATABASE IF NOT EXISTS card_txn")
spark.sql("USE card_txn")
spark.sql("DROP TABLE IF EXISTS transaction_features")
spark.sql(f"""
    CREATE EXTERNAL TABLE transaction_features
    USING PARQUET
    LOCATION '{abs_features_path}'
""")

print("Tables in Hive metastore (card_txn db):")
spark.sql("SHOW TABLES").show(truncate=False)

t0 = time.monotonic()
row_count = spark.sql("SELECT COUNT(*) AS n FROM transaction_features").collect()[0]["n"]
count_query_sec = time.monotonic() - t0
print(f"Row count via HiveQL: {row_count} ({count_query_sec:.2f}s)")

# Real HiveQL analytical query 1: fraud rate by merchant state, top 10 by volume
fraud_by_state = spark.sql("""
    SELECT
        Merchant_State,
        COUNT(*) AS txn_count,
        ROUND(AVG(Is_Fraud) * 100, 4) AS fraud_rate_pct
    FROM transaction_features
    WHERE Merchant_State IS NOT NULL
    GROUP BY Merchant_State
    ORDER BY txn_count DESC
    LIMIT 10
""")
print("\nTop 10 merchant states by transaction volume, with fraud rate:")
fraud_by_state.show(truncate=False)

# Real HiveQL analytical query 2: window function (RANK) directly in HiveQL,
# not the DataFrame API — ranking MCC categories by fraud rate among
# categories with meaningful volume.
mcc_fraud_ranked = spark.sql("""
    SELECT MCC, txn_count, fraud_rate_pct, fraud_rank
    FROM (
        SELECT
            MCC,
            COUNT(*) AS txn_count,
            ROUND(AVG(Is_Fraud) * 100, 4) AS fraud_rate_pct,
            RANK() OVER (ORDER BY AVG(Is_Fraud) DESC) AS fraud_rank
        FROM transaction_features
        GROUP BY MCC
        HAVING COUNT(*) >= 10000
    ) ranked
    ORDER BY fraud_rank
    LIMIT 10
""")
print("\nTop 10 highest-fraud-rate MCC categories (min 10,000 transactions), HiveQL RANK():")
mcc_fraud_ranked.show(truncate=False)

# Collect results for the README/resume
results = {
    "hive_metastore": "embedded Derby (local), Spark enableHiveSupport()",
    "table_type": "EXTERNAL (points at existing Parquet, no data copy)",
    "row_count": row_count,
    "count_query_seconds": round(count_query_sec, 2),
    "top_merchant_states_by_volume": [r.asDict() for r in fraud_by_state.collect()],
    "top_fraud_rate_mcc_categories": [r.asDict() for r in mcc_fraud_ranked.collect()],
}
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/hive_layer_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved to {RESULTS_DIR}/hive_layer_results.json")
spark.sql("SHOW TABLES").show(truncate=False)
