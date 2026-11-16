"""POST /api/predict

Body:  {"features": [30 numbers in the dataset's column order]}
Reply: the SVM's malignant probability and verdict, plus the K-Means cluster and the
       patient's position on the first two principal components.

The models are trained in the notebooks. export_model.py writes their parameters to
model.json, and this file redoes the prediction maths in plain Python so the function
needs no third-party packages.
"""
import json
import math
from http.server import BaseHTTPRequestHandler
from pathlib import Path

MODEL = json.loads((Path(__file__).parent / "model.json").read_text())
SVM, CLUSTER, N_FEATURES = MODEL["svm"], MODEL["cluster"], len(MODEL["features"])


def _standardise(values, mean, scale):
    return [(v - m) / s for v, m, s in zip(values, mean, scale)]


def svm_probability(values):
    """RBF-SVM decision function followed by Platt scaling, as scikit-learn's SVC does."""
    z = _standardise(values, SVM["mean"], SVM["scale"])
    decision = SVM["intercept"]
    for sv, coef in zip(SVM["support_vectors"], SVM["dual_coef"]):
        dist2 = sum((a - b) ** 2 for a, b in zip(sv, z))
        decision += coef * math.exp(-SVM["gamma"] * dist2)
    # libsvm fits the sigmoid on the negated decision value, giving P(benign).
    p_benign = 1.0 / (1.0 + math.exp(-decision * SVM["prob_a"] + SVM["prob_b"]))
    p_benign = min(max(p_benign, 1e-7), 1 - 1e-7)
    return _pairwise_coupling(p_benign)[1], decision


def _pairwise_coupling(r01):
    """libsvm's multiclass_probability for two classes.

    scikit-learn passes the pairwise sigmoid output through this iterative solver, which
    stops at a tolerance of 0.0025. Repeating it here keeps the result identical to
    SVC.predict_proba instead of merely close to it.
    """
    r = [[0.0, r01], [1.0 - r01, 0.0]]
    k, eps = 2, 0.005 / 2
    p = [0.5, 0.5]
    q = [[r[1][0] ** 2, -r[1][0] * r[0][1]], [-r[0][1] * r[1][0], r[0][1] ** 2]]
    for _ in range(100):
        qp = [sum(q[t][j] * p[j] for j in range(k)) for t in range(k)]
        pqp = sum(p[t] * qp[t] for t in range(k))
        if max(abs(qp[t] - pqp) for t in range(k)) < eps:
            break
        for t in range(k):
            diff = (-qp[t] + pqp) / q[t][t]
            p[t] += diff
            pqp = (pqp + diff * (diff * q[t][t] + 2 * qp[t])) / (1 + diff) / (1 + diff)
            for j in range(k):
                qp[j] = (qp[j] + diff * q[t][j]) / (1 + diff)
                p[j] /= 1 + diff
    return p


def cluster_of(values):
    """Standardise, project onto the principal components, pick the nearest K-Means centroid."""
    z = _standardise(values, CLUSTER["mean"], CLUSTER["scale"])
    centred = [a - b for a, b in zip(z, CLUSTER["pca_mean"])]
    coords = [sum(c * v for c, v in zip(comp, centred)) for comp in CLUSTER["components"]]
    dists = [math.sqrt(sum((a - b) ** 2 for a, b in zip(coords, centre))) for centre in CLUSTER["centroids"]]
    k = dists.index(min(dists))
    return k, coords, dists


def predict(values):
    probability, decision = svm_probability(values)
    k, coords, dists = cluster_of(values)
    malignant_like = bool(CLUSTER["cluster_is_malignant"][k])
    return {
        "probability": probability,
        "threshold": SVM["threshold"],
        "malignant": probability >= SVM["threshold"],
        "malignant_at_default": probability >= 0.5,
        "decision_value": decision,
        "cluster": {
            "index": k,
            "malignant_like": malignant_like,
            "name": "malignant-like" if malignant_like else "benign-like",
            "distances": dists,
        },
        "pca": coords[:2],
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 20000:
                return self._send(413, {"error": "Request body is too large."})
            values = json.loads(self.rfile.read(length) or b"{}").get("features")
        except (ValueError, AttributeError):
            return self._send(400, {"error": "Send JSON like {\"features\": [30 numbers]}."})
        if not isinstance(values, list) or len(values) != N_FEATURES:
            return self._send(400, {"error": f"'features' must be a list of {N_FEATURES} numbers."})
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            return self._send(400, {"error": "Every feature must be a finite number."})
        self._send(200, predict([float(v) for v in values]))

    def do_GET(self):
        self._send(200, {"model": "RBF-SVM + K-Means on PCA", "features": MODEL["features"],
                         "usage": "POST {\"features\": [30 numbers]} to this URL"})
