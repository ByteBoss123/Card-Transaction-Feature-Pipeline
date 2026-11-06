# Card Transaction Feature Pipeline: 24.4M Rows on Spark, EMR and Hive

A batch data pipeline that ingests 24.4M card transactions, engineers per-card window
features in PySpark, validates them, and publishes the feature table to a Hive metastore
(AWS Glue Data Catalog on EMR) for HiveQL analysis. A fraud classifier on the published
features is the downstream consumer and the end-to-end check that the features carry signal.

**The data is synthetic.** IBM TabFormer is a synthetic credit-card transaction dataset
generated to mimic real transaction-network statistics. No real cardholders, merchants or
transactions are involved, and results here say nothing about real-world fraud rates.

## Dataset

[IBM TabFormer](https://github.com/IBM/TabFormer), `card_transaction.v1.csv`: 24,386,900 rows,
2.35 GB, 6,139 cards, binary `Is Fraud?` label (29,757 fraud rows, 0.12%). License
CDLA-Sharing-1.0. Not redistributed here; download from the TabFormer repo and place it in
`./data/` (or set `TXN_DATA_DIR`).

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Ingest | `src/01_ingest.py` | typed, cleaned Parquet (`parquet/clean`, 366 MB on S3) |
| 2. Features | `src/02_features.py` | per-card velocity / rolling-amount / merchant features via Spark window functions over the full 24.4M rows (`parquet/features`, 1.36 GB, 33 files on S3) |
| 3. Validate | `src/03_validate.py` | row counts, spot checks, null placement |
| 4. Export | `src/04_export_sample.py` | stratified model sample, all fraud + 15:1 non-fraud (`parquet/model_sample`) |
| 5. Model | `src/05_train_model.py` | XGBoost on the sample, leaky vs leakage-corrected comparison |
| 6. Hive layer | `src/06_hive_layer.py` | `card_txn.transaction_features` as an EXTERNAL Parquet table + HiveQL analytics |

Each script reads `TXN_DATA_DIR`, `TXN_OUTPUT_DIR`, `TXN_WAREHOUSE_DIR` and
`TXN_USE_GLUE_CATALOG`, so the same code runs locally (Derby metastore) and on EMR
(S3 paths, Glue catalog).

## Run on AWS EMR

`emr_deploy/01_upload_to_s3.sh` stages code and data in S3; `emr_deploy/02_create_cluster.sh`
creates an EMR cluster and submits steps 1-4 and 6 as `spark-submit --deploy-mode cluster`
jobs (`emr_steps.json`), with logs to S3 and the cluster terminating afterwards.

Evidence of the cluster run, checked against the AWS account on 2026-11-06:
- Step logs for EMR clusters `j-09675982OLLLGPMEJKW` and `j-03797882Z1B9A1BPQ4DP`
  (2026-08-26, 5 steps on the second cluster) in the project's S3 log prefix; driver output and the
  generated step file are in `run_logs/`.
- Pipeline outputs in S3: `parquet/clean` (19 files, 366 MB), `parquet/features`
  (33 files, 1.36 GB), `parquet/model_sample` (29 files, 20.6 MB).
- Glue Data Catalog table `transaction_features`, created 2026-08-26 02:45 UTC, pointing at the
  `parquet/features` output in S3.

Model training (step 5) ran locally on the exported sample, not on the cluster.

## Hive layer results

From `run_logs/hive_real_stdout`, HiveQL over the registered 24,386,900-row table:
- `COUNT(*)` 24,386,900 in 2.75 s.
- Highest-volume merchant states: CA 2.59M transactions (0.029% fraud), TX 1.79M (0.017%),
  FL 1.46M (0.022%).
- Highest-fraud MCC by `RANK() OVER (ORDER BY fraud_rate DESC)`, min 10,000 transactions:
  MCC 5732 at 6.69%.

## Model results (downstream check)

Stratified 476,080-row sample (all 29,757 fraud + 15:1 non-fraud), 80/20 stratified split,
held-out test set (`results/results_full_run.json`):

| Feature set | ROC-AUC | PR-AUC |
|---|---|---|
| Includes `merchant_fraud_rate` | 0.9957 | 0.9678 |
| **Leakage-excluded (reported)** | **0.9871** | **0.9256** |

`merchant_fraud_rate` was computed with a `groupBy` over the whole dataset, test rows
included, so it leaks labels into training. It took 55% of feature importance and inflated
PR-AUC by 0.04. The reported numbers exclude it.

PR-AUC is measured on the 15:1 sample (6.25% fraud), not the true 0.12% base rate, so it
overstates precision at production prevalence.

## Naming

The project was renamed after the EMR run. In `run_logs/`, names, paths and the S3 bucket were
updated to match; the run output is otherwise unchanged.

## Limits

- Synthetic data (see top). Merchant names are synthetic identifiers.
- The classifier trains on a stratified sample, not all 24.4M rows; the Spark stages run on all rows.
- The EMR run is evidenced by S3 logs, outputs and the Glue table; the EMR cluster records
  themselves have aged out of the EMR console.

## Stack

PySpark, Spark SQL window functions, Hive metastore (Derby locally, AWS Glue Data Catalog on
EMR), AWS EMR, S3, XGBoost, scikit-learn, pandas.

## License

Code: MIT (see `LICENSE`). Dataset: CDLA-Sharing-1.0 (IBM TabFormer), not redistributed.
