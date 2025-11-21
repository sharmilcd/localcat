import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import os

FEATURES = ["token_count","has_digit","contains_co","amount_log"]
MODEL_PATH = "models/lightgbm_baseline.pkl"

def load_data(path):
    return pd.read_parquet(path)

def prepare(df):
    df = df.copy()
    if "category" not in df.columns:
        raise ValueError("Dataset must include 'category' column")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])
    X = df[FEATURES].fillna(0.0)
    y = df["label"]
    return X, y, le

def train(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "verbosity": -1,
        "seed": 42
    }
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=500,
        early_stopping_rounds=30,
        verbose_eval=50
    )
    return model

def evaluate(model, X, y, le):
    preds = model.predict(X)
    y_pred = preds.argmax(axis=1)
    macro = f1_score(y, y_pred, average="macro")
    print("Macro F1:", macro)
    print(classification_report(y, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y, y_pred)
    return macro, cm

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=MODEL_PATH)
    args = p.parse_args()
    df = load_data(args.data)
    X, y, le = prepare(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = train(X_train, y_train, X_val, y_val)
    mac, cm = evaluate(model, X_val, y_val, le)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model":model, "label_encoder":le}, args.out)
    print("Saved model to", args.out)
