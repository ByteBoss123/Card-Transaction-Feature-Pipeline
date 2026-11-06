"""
Step 5: Train an XGBoost fraud classifier on the Spark-engineered features.

Trains and reports TWO versions on purpose:
  1. Full feature set (includes merchant_fraud_rate) — inflated by target leakage.
  2. Leakage-excluded feature set (drops merchant_fraud_rate, merchant_txn_count).

The second is the headline result reported in the README. The first is kept and
printed so the leakage's actual effect on the metric is visible and quantified,
not hidden.
"""
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import xgboost as xgb

OUTPUT_DIR = os.environ.get("TXN_OUTPUT_DIR", "./output/parquet")
RESULTS_PATH = os.environ.get("TXN_RESULTS_PATH", "./output/results.json")

df = pd.read_parquet(f"{OUTPUT_DIR}/model_sample")
print("Loaded shape:", df.shape)
print("Fraud rate in sample:", df["Is_Fraud"].mean())

cat_cols = ["Use_Chip", "MCC", "Merchant_State"]
for c in cat_cols:
    df[c] = df[c].astype("category")

df = df.drop(columns=["Zip"])  # too sparse/high-cardinality; dropped rather than fabricating a treatment

num_cols = [c for c in df.columns if c not in cat_cols + ["Is_Fraud"]]
for c in num_cols:
    df[c] = df[c].fillna(-1)

X = df.drop(columns=["Is_Fraud"])
y = df["Is_Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def fit_and_eval(X_tr, X_te, label):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", enable_categorical=True, tree_method="hist", random_state=42,
    )
    model.fit(X_tr, y_train, eval_set=[(X_te, y_test)], verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"\n=== {label} ===")
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print(classification_report(y_test, pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    importances = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    print("Feature importances:\n", importances)
    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "importances": importances.to_dict()}


results_full = fit_and_eval(X_train, X_test, "FULL FEATURE SET (includes merchant_fraud_rate — leakage present)")

leak_cols = ["merchant_fraud_rate", "merchant_txn_count"]
results_clean = fit_and_eval(
    X_train.drop(columns=leak_cols), X_test.drop(columns=leak_cols),
    "LEAKAGE-EXCLUDED FEATURE SET (headline / reported result)"
)

print(f"\nLeakage inflation: PR-AUC {results_full['pr_auc']:.4f} (leaky) vs "
      f"{results_clean['pr_auc']:.4f} (clean) — leaky version overstates PR-AUC by "
      f"{results_full['pr_auc'] - results_clean['pr_auc']:.4f}")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "sample_shape": list(df.shape),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "test_fraud_rate": float(y_test.mean()),
        "full_features": results_full,
        "leakage_excluded_features": results_clean,
    }, f, indent=2, default=str)

print(f"\nSaved {RESULTS_PATH}")
