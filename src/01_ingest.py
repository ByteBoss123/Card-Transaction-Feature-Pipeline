"""
Step 1: Ingest raw TabFormer CSV, clean/typecast columns, write to compressed Parquet.

Dataset: IBM TabFormer synthetic credit card transactions (24.4M rows, 12 fields + label).
Download separately (not included in this repo — see README): 
    https://github.com/IBM/TabFormer/tree/main/data/credit_card
Place the extracted card_transaction.v1.csv under DATA_DIR before running.
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

DATA_DIR = os.environ.get("TXN_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("TXN_OUTPUT_DIR", "./output/parquet")

spark = (
    SparkSession.builder
    .appName("CardTxn-Ingest")
    .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g"))
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("User", IntegerType()),
    StructField("Card", IntegerType()),
    StructField("Year", IntegerType()),
    StructField("Month", IntegerType()),
    StructField("Day", IntegerType()),
    StructField("Time", StringType()),
    StructField("Amount", StringType()),
    StructField("Use_Chip", StringType()),
    StructField("Merchant_Name", StringType()),
    StructField("Merchant_City", StringType()),
    StructField("Merchant_State", StringType()),
    StructField("Zip", StringType()),
    StructField("MCC", StringType()),
    StructField("Errors", StringType()),
    StructField("Is_Fraud", StringType()),
])

raw = (
    spark.read
    .option("header", True)
    .schema(schema)
    .csv(f"{DATA_DIR}/card_transaction.v1.csv")
)

print("Raw row count:", raw.count())

clean = (
    raw
    .withColumn("Amount", F.regexp_replace("Amount", r"[\$,]", "").cast("double"))
    .withColumn("Is_Fraud", (F.col("Is_Fraud") == "Yes").cast("int"))
    .withColumn("Has_Error", (F.col("Errors").isNotNull()).cast("int"))
    .withColumn(
        "event_ts",
        F.to_timestamp(
            F.concat_ws(" ",
                F.concat_ws("-", F.col("Year").cast("string"), F.lpad(F.col("Month").cast("string"), 2, "0"), F.lpad(F.col("Day").cast("string"), 2, "0")),
                F.col("Time")
            ),
            "yyyy-MM-dd HH:mm"
        )
    )
    .withColumn("card_id", F.concat_ws("_", F.col("User").cast("string"), F.col("Card").cast("string")))
    .drop("Errors")
)

clean.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/clean")

print("Ingest complete.")
print("Fraud rate:", clean.selectExpr("avg(Is_Fraud) as rate").collect()[0]["rate"])
