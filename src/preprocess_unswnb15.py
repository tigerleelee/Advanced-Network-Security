"""
SEC3044 - Member 2: Data & Preprocessing (UNSW-NB15)

Run:  python preprocess_unswnb15.py
Needs: pip install pandas numpy scikit-learn joblib pyarrow

The dataset already comes split into training-set / testing-set,
so we keep that split (no new train_test_split).

Outputs (in OUT_DIR):
  train.parquet, test.parquet, scaler.joblib, features.json,
  class_weights.json, feature_importance.csv, preprocessing_log.txt
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- CHANGE THIS ----------------
DATA_DIR = r"C:\Users\Owner\Downloads\archive (1)"
OUT_DIR = r"C:\Users\Owner\Documents\UNSW_output"
# ---------------------------------------------
RANDOM_STATE = 42
CORR_THRESHOLD = 0.95
TOP_K = 30
SAMPLE_N = 200_000
MIN_CAT_COUNT = 1000    # categories rarer than this become "other"

os.makedirs(OUT_DIR, exist_ok=True)
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ============ STEP 1: Load ============
tr = pd.read_parquet(os.path.join(DATA_DIR, "UNSW_NB15_training-set.parquet"))
te = pd.read_parquet(os.path.join(DATA_DIR, "UNSW_NB15_testing-set.parquet"))
log(f"[1] Loaded train: {len(tr):,} rows | test: {len(te):,} rows | {tr.shape[1]} columns")

# category columns -> plain text so they are easy to handle
for d in (tr, te):
    for c in ("proto", "service", "state", "attack_cat"):
        d[c] = d[c].astype(str)

# ============ STEP 2: Clean ============
log(f"[2] NaN values in train: {int(tr.isna().sum().sum()):,} | test: {int(te.isna().sum().sum()):,}")

# 2a. duplicates inside each set
n_tr, n_te = len(tr), len(te)
tr = tr.drop_duplicates().reset_index(drop=True)
te = te.drop_duplicates().reset_index(drop=True)
log(f"[2] Duplicates removed -> train: {n_tr - len(tr):,} (left {len(tr):,}) | "
    f"test: {n_te - len(te):,} (left {len(te):,})")

# 2b. test rows that also appear in train (data leakage)
train_hashes = set(pd.util.hash_pandas_object(tr, index=False))
leak = pd.util.hash_pandas_object(te, index=False).isin(train_hashes)
log(f"[2] Test rows identical to a train row (removed): {int(leak.sum()):,}")
te = te[~leak].reset_index(drop=True)
log(f"[2] Final test size: {len(te):,}")

# 2c. drop TCP sequence numbers (random-looking, not useful for detection)
tr = tr.drop(columns=["stcpb", "dtcpb"])
te = te.drop(columns=["stcpb", "dtcpb"])

# 2d. labels: use the same names as the CIC-IDS2017 files
for d in (tr, te):
    d["Label"] = d["attack_cat"].replace({"Normal": "BENIGN"})
    d["Label_binary"] = d["label"].astype("int8")
    d.drop(columns=["attack_cat", "label"], inplace=True)

log("[2] Label distribution (train):")
log(tr["Label"].value_counts().to_string())
log("[2] Label distribution (test):")
log(te["Label"].value_counts().to_string())

# ============ STEP 3: Encode text columns ============
cat_cols = ["proto", "service", "state"]
for c in cat_cols:
    counts = tr[c].value_counts()
    keep = counts[counts >= MIN_CAT_COUNT].index
    for d in (tr, te):
        d[c] = d[c].where(d[c].isin(keep), "other")
    log(f"[3] {c}: kept {list(keep)} + other")

n_train = len(tr)
all_df = pd.concat([tr, te], ignore_index=True)
all_df = pd.get_dummies(all_df, columns=cat_cols, dtype="float32")
tr = all_df.iloc[:n_train].copy()
te = all_df.iloc[n_train:].copy()

label_cols = ["Label", "Label_binary"]
feature_cols = [c for c in tr.columns if c not in label_cols]
tr[feature_cols] = tr[feature_cols].astype("float32")
te[feature_cols] = te[feature_cols].astype("float32")
log(f"[3] Features after encoding: {len(feature_cols)}")

# ============ STEP 4: Feature selection (TRAIN only) ============
nunique = tr[feature_cols].nunique()
constant = nunique[nunique <= 1].index.tolist()
selected = [c for c in feature_cols if c not in constant]
log(f"[4] Constant features removed ({len(constant)}): {constant}")

sample = tr.sample(min(SAMPLE_N, len(tr)), random_state=RANDOM_STATE)
corr = sample[selected].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
selected = [c for c in selected if c not in high_corr]
log(f"[4] Highly correlated features removed ({len(high_corr)}): {high_corr}")
log(f"[4] {len(selected)} features left")

rf = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
)
rf.fit(sample[selected], sample["Label"])
imp = pd.Series(rf.feature_importances_, index=selected).sort_values(ascending=False)
final_features = imp.head(TOP_K).index.tolist()
log(f"[4] Top {len(final_features)} features by Random Forest importance:")
log(imp.head(TOP_K).round(4).to_string())
imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))

# ============ STEP 5: Standardise (fit on TRAIN only) ============
scaler = StandardScaler()
X_train = scaler.fit_transform(tr[final_features])
X_test = scaler.transform(te[final_features])

train_out = pd.DataFrame(X_train, columns=final_features, dtype="float32")
train_out["Label"] = tr["Label"].values
train_out["Label_binary"] = tr["Label_binary"].values
test_out = pd.DataFrame(X_test, columns=final_features, dtype="float32")
test_out["Label"] = te["Label"].values
test_out["Label_binary"] = te["Label_binary"].values

# ============ STEP 6: Class weights ============
classes = np.sort(train_out["Label"].unique())
counts = train_out["Label"].value_counts()
class_weights = {c: float(len(train_out) / (len(classes) * counts[c])) for c in classes}
log("[6] Class weights (balanced):")
for c, w in class_weights.items():
    log(f"    {c}: {w:.3f}")

# ============ STEP 7: Save ============
train_out.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
test_out.to_parquet(os.path.join(OUT_DIR, "test.parquet"), index=False)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
with open(os.path.join(OUT_DIR, "features.json"), "w") as f:
    json.dump(final_features, f, indent=2)
with open(os.path.join(OUT_DIR, "class_weights.json"), "w") as f:
    json.dump(class_weights, f, indent=2)
with open(os.path.join(OUT_DIR, "preprocessing_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log(f"[7] Saved all files to {OUT_DIR}")