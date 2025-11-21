import re
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import unicodedata
from Levenshtein import distance as lev

ALIAS_PATH = "data/merchant_aliases.json"

def load_aliases(path=ALIAS_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9\s\-/&\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_domain(s):
    # If s looks like a URL, extract domain, else attempt to split tokens with dot
    if "http" in s or "." in s:
        try:
            parsed = urlparse(s if s.startswith("http") else "http://"+s)
            return parsed.netloc.split(":")[0].lower()
        except:
            return ""
    return ""

def merchant_normalize(name, aliases):
    n = normalize_text(name)
    # exact aliases
    if n in aliases:
        return aliases[n]
    # fuzzy match a few top aliases by Levenshtein
    best, best_dist = None, 999
    for k in aliases:
        d = lev(n, k)
        if d < best_dist and d <= 2:
            best_dist, best = d, aliases[k]
    return best if best else n

def feature_engineer(df, alias_path=ALIAS_PATH):
    aliases = load_aliases(alias_path)
    df = df.copy()
    df["raw"] = df["raw"].fillna("")
    df["raw_norm"] = df["raw"].apply(normalize_text)
    df["domain"] = df["raw"].apply(extract_domain)
    df["merchant"] = df["raw"].apply(lambda x: merchant_normalize(x, aliases))
    # ngram simple features
    df["token_count"] = df["raw_norm"].apply(lambda s: len(s.split()))
    df["has_digit"] = df["raw_norm"].str.contains(r"\d").astype(int)
    df["contains_co"] = df["raw_norm"].str.contains("co|company|ltd|inc").astype(int)
    # amount features (if amount col exists)
    if "amount" in df.columns:
        df["amount_log"] = np.log1p(np.abs(df["amount"]))
    else:
        df["amount_log"] = 0.0
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    df2 = feature_engineer(df)
    df2.to_parquet(args.output, index=False)
    print("Preprocessed:", args.output)
