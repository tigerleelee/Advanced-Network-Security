import json, time
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def load(folder):
    feats = json.load(open(folder + '/features.json'))
    tr = pd.read_parquet(folder + '/train.parquet')
    te = pd.read_parquet(folder + '/test.parquet')
    return (tr[feats].values.astype('float32'), tr['Label_binary'].values,
            te[feats].values.astype('float32'), te['Label_binary'].values, feats)

def evaluate(name, proba_fn, X, y, thr=0.5):
    t0 = time.perf_counter()
    p = proba_fn(X)                       # probability of attack, 1-D array
    us_per_flow = (time.perf_counter() - t0) / len(X) * 1e6
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    row = dict(model=name, accuracy=accuracy_score(y, pred),
               precision=precision_score(y, pred), recall=recall_score(y, pred),
               f1=f1_score(y, pred), roc_auc=roc_auc_score(y, p),
               fpr=fp / (fp + tn), fnr=fn / (fn + tp), infer_us=us_per_flow)
    return row, p