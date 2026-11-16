"""Export the trained models to plain JSON for the web app.

The web backend runs without scikit-learn, so this script writes out the numbers the
models need at prediction time and checks that the pure-Python maths in api/predict.py
reproduces scikit-learn's output.

Run from the repository root after the notebooks:

    python webapp/export_model.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "webapp"
RANDOM_STATE = 42

# ---------- Program 1: the tuned SVM pipeline ----------
bundle = joblib.load(ROOT / "models" / "final_model.joblib")
pipe, threshold, features = bundle["model"], bundle["threshold"], bundle["features"]
scaler, svc = pipe.named_steps["scaler"], pipe.named_steps["model"]

svm = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "support_vectors": svc.support_vectors_.tolist(),
    "dual_coef": svc.dual_coef_[0].tolist(),
    "intercept": float(svc.intercept_[0]),
    "gamma": float(svc._gamma),
    "prob_a": float(svc.probA_[0]),
    "prob_b": float(svc.probB_[0]),
    "threshold": float(threshold),
}

# ---------- Program 2: scaler -> PCA(90%) -> K-Means, refitted exactly as in notebook 03 ----------
df = pd.read_csv(ROOT / "data" / "breast_cancer.csv")
X, y = df[features].values, df["target"].values
scaler_all = StandardScaler().fit(X)
X_scaled = scaler_all.transform(X)
n90 = int(np.argmax(np.cumsum(PCA(random_state=RANDOM_STATE).fit(X_scaled).explained_variance_ratio_) >= 0.90) + 1)
pca = PCA(n_components=n90, random_state=RANDOM_STATE).fit(X_scaled)
X_red = pca.transform(X_scaled)
km = KMeans(n_clusters=2, n_init=50, random_state=RANDOM_STATE).fit(X_red)
majority = {int(c): int(np.bincount(y[km.labels_ == c]).argmax()) for c in range(2)}
agree = int(sum(majority[int(c)] == t for c, t in zip(km.labels_, y)))
assert agree == 518, f"K-Means agreement changed: {agree}"

cluster = {
    "mean": scaler_all.mean_.tolist(),
    "scale": scaler_all.scale_.tolist(),
    "pca_mean": pca.mean_.tolist(),
    "components": pca.components_.tolist(),
    "centroids": km.cluster_centers_.tolist(),
    "cluster_is_malignant": [majority[0], majority[1]],
}

(WEB / "api").mkdir(parents=True, exist_ok=True)
(WEB / "api" / "model.json").write_text(json.dumps({"features": features, "svm": svm, "cluster": cluster}))

# ---------- Data for the page: the PCA map and the 114 test patients ----------
test = pd.read_csv(ROOT / "data" / "test.csv")
points = [[round(float(a), 3), round(float(b), 3), int(t), majority[int(c)]]
          for (a, b), t, c in zip(X_red[:, :2], y, km.labels_)]
patients = [{"id": i + 1, "actual": int(r["target"]), "values": [float(r[f]) for f in features]}
            for i, (_, r) in enumerate(test.iterrows())]
ranges = [{"min": float(df[f].min()), "max": float(df[f].max())} for f in features]
(WEB / "data.json").write_text(json.dumps({"features": features, "ranges": ranges, "points": points, "patients": patients}))

# ---------- Check: the pure-Python backend must match scikit-learn ----------
sys.path.insert(0, str(WEB / "api"))
import predict as backend  # noqa: E402

proba = pipe.predict_proba(test[features])[:, 1]
worst = 0.0
for row, p, c in zip(test[features].values, proba, km.predict(pca.transform(scaler_all.transform(test[features].values)))):
    out = backend.predict(row.tolist())
    worst = max(worst, abs(out["probability"] - p))
    assert out["cluster"]["malignant_like"] == bool(majority[int(c)])
assert worst < 1e-6, f"backend differs from scikit-learn by {worst}"
print(f"exported {len(svm['support_vectors'])} support vectors, {n90} PCA components")
print(f"backend matches scikit-learn on {len(test)} test patients (max probability difference {worst:.2e})")
