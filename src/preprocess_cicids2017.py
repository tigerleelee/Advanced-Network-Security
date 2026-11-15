"""
SEC3044 - Member 2: Data & Preprocessing (CIC-IDS2017)
 
Run:  python preprocess_cicids2017.py
Needs: pip install pandas numpy scikit-learn joblib pyarrow
 
Outputs (in OUT_DIR):
  train.parquet, test.parquet   -> give to Member 3
  scaler.joblib                 -> give to Member 3 AND Member 4
  features.json                 -> give to Member 3 AND Member 4
  class_weights.json            -> give to Member 3
  preprocessing_log.txt         -> numbers for your report
"""
import glob
import json
import os
 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
# ---------------- CHANGE THIS ----------------
DATA_DIR = r"C:\Users\Owner\Downloads\archive\MachineLearningCSV\MachineLearningCVE"
OUT_DIR = r"C:\Users\Owner\Documents\NS_output"
# ---------------------------------------------
RANDOM_STATE = 42       # fixed so everyone gets the same result
TEST_SIZE = 0.30        # 70/30 split
CORR_THRESHOLD = 0.95   # drop one of two features if correlation > this
TOP_K = 30              # number of final features
SAMPLE_N = 200_000      # sample size for correlation / RF importance (saves time)
 
os.makedirs(OUT_DIR, exist_ok=True)
log_lines = []
 
 
def log(msg):
    print(msg)
    log_lines.append(str(msg))
 
 
# ============ STEP 1: Load and merge the 8 CSV files ============
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
log(f"[1] Found {len(files)} CSV files")
frames = []
for f in files:
    # latin1 avoids the encoding error in the Thursday file
    d = pd.read_csv(f, encoding="latin1", low_memory=False)
    d.columns = d.columns.str.strip()          # remove spaces in column names
    log(f"    {os.path.basename(f)}: {len(d):,} rows")
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
del frames
log(f"[1] Merged: {df.shape[0]:,} rows, {df.shape[1]} columns")
 
# ============ STEP 2: Clean ============
# 2a. fix labels (Web Attack labels contain a strange character)
df["Label"] = df["Label"].str.replace(r"[^\x20-\x7E]+", "-", regex=True).str.strip()
 
# 2b. replace inf with NaN, then drop rows with NaN
feature_cols = [c for c in df.columns if c != "Label"]
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
n_inf = int(np.isinf(df[feature_cols].to_numpy(dtype="float64")).sum())
log(f"[2] inf values found: {n_inf:,}")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
log(f"[2] NaN values found: {int(df[feature_cols].isna().sum().sum()):,}")
before = len(df)
df.dropna(inplace=True)
log(f"[2] Rows after dropping NaN/inf: {len(df):,} (removed {before - len(df):,})")
 
# 2c. remove duplicates (BEFORE splitting)
before = len(df)
df.drop_duplicates(inplace=True)
log(f"[2] Rows after removing duplicates: {len(df):,} (removed {before - len(df):,})")
 
# 2d. shrink memory
df[feature_cols] = df[feature_cols].astype("float32")
 
log("[2] Label distribution after cleaning:")
log(df["Label"].value_counts().to_string())
 
# ============ STEP 3: Train/test split (stratified) ============
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df["Label"], random_state=RANDOM_STATE
)
del df
log(f"[3] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
 
# ============ STEP 4: Feature selection (using TRAIN only) ============
# 4a. drop constant columns
nunique = train_df[feature_cols].nunique()
constant = nunique[nunique <= 1].index.tolist()
selected = [c for c in feature_cols if c not in constant]
log(f"[4] Constant features removed ({len(constant)}): {constant}")
 
# 4b. drop highly correlated features
sample = train_df.sample(min(SAMPLE_N, len(train_df)), random_state=RANDOM_STATE)
corr = sample[selected].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
selected = [c for c in selected if c not in high_corr]
log(f"[4] Highly correlated features removed ({len(high_corr)}); {len(selected)} left")
 
# 4c. Random Forest importance -> keep top K
rf = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
)
rf.fit(sample[selected], sample["Label"])
imp = pd.Series(rf.feature_importances_, index=selected).sort_values(ascending=False)
final_features = imp.head(TOP_K).index.tolist()
log(f"[4] Top {TOP_K} features by Random Forest importance:")
log(imp.head(TOP_K).round(4).to_string())
imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
 
# ============ STEP 5: Standardise (fit on TRAIN only) ============
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[final_features])
X_test = scaler.transform(test_df[final_features])
 
train_out = pd.DataFrame(X_train, columns=final_features, dtype="float32")
train_out["Label"] = train_df["Label"].values
test_out = pd.DataFrame(X_test, columns=final_features, dtype="float32")
test_out["Label"] = test_df["Label"].values
 
# binary label too (0 = BENIGN, 1 = attack)
for d in (train_out, test_out):
    d["Label_binary"] = (d["Label"] != "BENIGN").astype("int8")
 
# ============ STEP 6: Class imbalance -> class weights ============
# (Apply SMOTE only on the TRAIN set if you really need it; never on test.)
classes = np.sort(train_out["Label"].unique())
counts = train_out["Label"].value_counts()
n_total, n_classes = len(train_out), len(classes)
class_weights = {c: float(n_total / (n_classes * counts[c])) for c in classes}
log("[6] Class weights (balanced):")
for c, w in class_weights.items():
    log(f"    {c}: {w:.3f}")
 
# ============ STEP 7: Save everything ============
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