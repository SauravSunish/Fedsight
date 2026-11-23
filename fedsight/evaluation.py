"""Leakage-checked model comparison, separate calibration and held-out evaluation.

Models are persisted as numerical JSON; no untrusted pickle/joblib deserialization.
"""
from __future__ import annotations
import math
import numpy as np
from .core import FEATURE_VERSION, canonical, digest, uid
from .pair_features import FEATURE_NAMES

SPLITS = ("train", "calibration", "validation", "test")


def validate_dataset(rows: list[dict]):
    if not rows or len(rows) > 100_000:
        raise ValueError("Provide 1–100,000 labelled pairs.")
    seen = {kind: {} for kind in ("identity", "sequence", "case")}
    counts = {s: [0, 0] for s in SPLITS}
    pair_keys = set()
    if len({bool(r.get("synthetic", False)) for r in rows}) > 1:
        raise ValueError("Do not mix synthetic and real evaluation records.")
    for row in rows:
        split = row.get("split")
        if split not in SPLITS or row.get("label") not in (0, 1):
            raise ValueError("Every pair needs split=train/calibration/validation/test and binary label.")
        if row.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Pair-feature schema version mismatch.")
        values = np.asarray([row["features"][key] for key in FEATURE_NAMES], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("All feature values must be finite.")
        for kind, fields in (("identity", ("identity_a", "identity_b")), ("sequence", ("sequence_a", "sequence_b")), ("case", ("case_id",))):
            for field in fields:
                token = str(row.get(field, "")).strip()
                if not isinstance(row.get(field), str) or not token:
                    raise ValueError(f"Missing leakage-control field: {field}.")
                if token in seen[kind] and seen[kind][token] != split:
                    raise ValueError(f"{kind} leakage across splits: {token}.")
                seen[kind][token] = split
        if row["sequence_a"] == row["sequence_b"]:
            raise ValueError("Training pairs must span distinct source sequences.")
        if int(row["identity_a"] == row["identity_b"]) != row["label"]:
            raise ValueError("Identity labels contradict pair labels.")
        key = (row["sequence_a"], row["sequence_b"], row["identity_a"], row["identity_b"])
        reverse = (key[1], key[0], key[3], key[2])
        if key in pair_keys or reverse in pair_keys:
            raise ValueError("Duplicate or reversed labelled pair.")
        pair_keys.add(key)
        counts[split][row["label"]] += 1
    if any(min(c) < 2 for c in counts.values()):
        raise ValueError("Each split needs at least two positive and two negative examples.")
    return counts


def binary_metrics(labels, probabilities, threshold=.5):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if not len(y) or y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Metrics require aligned binary labels and finite probabilities.")
    pred = p >= threshold
    tp, fp, fn, tn = [int(v.sum()) for v in (pred & (y == 1), pred & (y == 0), ~pred & (y == 1), ~pred & (y == 0))]
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    bins, ece = [], 0.0
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1)
        if mask.any():
            confidence, accuracy, count = float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())
            ece += count / len(y) * abs(confidence - accuracy)
            bins.append({"mean_probability": confidence, "observed_fraction": accuracy, "count": count})
    return {"n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12), "false_accept_rate": fp / max(fp + tn, 1),
            "false_reject_rate": fn / max(fn + tp, 1), "brier": float(np.mean((p - y) ** 2)), "ece": ece,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None, "reliability_bins": bins}


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _boosted_stumps(x, y, weights, rounds=60, rate=.1):
    """Newton boosting of shallow decision trees using weighted binary log loss.

    Implemented in NumPy so no optional ensemble extension is required. All
    candidate cuts and leaf values are learned from training rows only.
    """
    prior = float(np.average(y, weights=weights))
    initial = math.log(prior / (1 - prior))
    logits = np.full(len(y), initial)
    splits = []
    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind="stable")
        values = x[order, feature]
        ends = np.flatnonzero(values[:-1] < values[1:])
        ends = ends[(ends >= 1) & (ends < len(y) - 2)]
        if len(ends) > 64:
            ends = ends[np.linspace(0, len(ends) - 1, 64, dtype=int)]
        if len(ends):
            splits.append((feature, order, ends))
    trees = []
    for _ in range(rounds):
        p = _sigmoid(logits)
        g, h = weights * (y - p), weights * p * (1 - p)
        g_total, h_total = g.sum(), h.sum()
        best_gain, best = 0., None
        for feature, order, ends in splits:
            gl, hl = np.cumsum(g[order])[ends], np.cumsum(h[order])[ends]
            gr, hr = g_total - gl, h_total - hl
            gains = gl**2 / (hl + 1) + gr**2 / (hr + 1) - g_total**2 / (h_total + 1)
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                cut = ends[j]
                threshold = (x[order[cut], feature] + x[order[cut + 1], feature]) / 2
                best_gain, best = gains[j], (feature, float(threshold), float(gl[j] / (hl[j] + 1)), float(gr[j] / (hr[j] + 1)))
        if best is None:
            break
        feature, threshold, left, right = best
        logits += rate * np.where(x[:, feature] <= threshold, left, right)
        trees.append({"left": [1, -1, -1], "right": [2, -1, -1], "feature": [feature, -2, -2],
                      "threshold": [threshold, -2., -2.], "value": [0., left, right]})
    return {"type": "gradient_boosting", "implementation": "numpy-newton-stumps-v1", "trees": trees,
            "learning_rate": rate, "intercept": initial}


def _serialize_model(model, name):
    if name == "logistic":
        return {"type": name, "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
    if name == "mlp":
        return {"type": name, "coefs": [v.tolist() for v in model.coefs_], "intercepts": [v.tolist() for v in model.intercepts_]}
    trees = []
    for estimator in model.estimators_[:, 0]:
        t = estimator.tree_
        trees.append({"left": t.children_left.tolist(), "right": t.children_right.tolist(), "feature": t.feature.tolist(),
                      "threshold": t.threshold.tolist(), "value": t.value[:, 0, 0].tolist()})
    prior = float(model.init_.class_prior_[1])
    return {"type": name, "trees": trees, "learning_rate": model.learning_rate, "intercept": math.log(prior / (1 - prior))}


def _raw(model, x):
    if model["type"] == "logistic":
        return x @ np.asarray(model["coef"]) + model["intercept"]
    if model["type"] == "mlp":
        value = x
        for i, (w, b) in enumerate(zip(model["coefs"], model["intercepts"])):
            value = value @ np.asarray(w) + np.asarray(b)
            if i < len(model["coefs"]) - 1:
                value = np.maximum(value, 0)
        return value.ravel()
    if model["type"] != "gradient_boosting":
        raise ValueError("Unsupported numerical model type.")
    result = np.full(len(x), model["intercept"], dtype=float)
    for tree in model["trees"]:
        for row, features in enumerate(x):
            node = 0
            for _ in range(len(tree["left"]) + 1):
                if tree["left"][node] == -1:
                    result[row] += model["learning_rate"] * tree["value"][node]
                    break
                node = tree["left"][node] if features[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
            else:
                raise ValueError("Invalid tree cycle.")
    return result


def predict_artifact(artifact: dict, feature_rows: list[dict]):
    if artifact.get("feature_version") != FEATURE_VERSION or artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Classifier feature schema does not match this application.")
    x = np.asarray([[row[k] for k in FEATURE_NAMES] for row in feature_rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Classifier inputs must be finite.")
    x = (x - np.asarray(artifact["mean"])) / np.asarray(artifact["scale"])
    raw = _raw(artifact["model"], x)
    return _sigmoid(raw * artifact["calibration"]["coef"] + artifact["calibration"]["intercept"])


def train_compare(rows: list[dict], seed=42, target_precision=.95):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    counts = validate_dataset(rows)
    if not 0 < target_precision <= 1:
        raise ValueError("Target precision must be in (0, 1].")
    data = {}
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        data[split] = (np.asarray([[r["features"][k] for k in FEATURE_NAMES] for r in selected]), np.asarray([r["label"] for r in selected]))
    scaler = StandardScaler().fit(data["train"][0])
    models = {"logistic": LogisticRegression(max_iter=1000, random_state=seed),
              "gradient_boosting": None,
              "mlp": MLPClassifier(hidden_layer_sizes=(24, 12), max_iter=500, random_state=seed)}
    comparisons, artifacts = [], {}
    for name, model in models.items():
        x, y = data["train"]
        fit_x, weights = scaler.transform(x), compute_sample_weight("balanced", y)
        if name == "gradient_boosting":
            spec = _boosted_stumps(fit_x, y, weights)
        else:
            model.fit(fit_x, y, sample_weight=weights)
            spec = _serialize_model(model, name)
        calibration_raw = _raw(spec, scaler.transform(data["calibration"][0]))
        cal = LogisticRegression(C=10, max_iter=1000).fit(calibration_raw[:, None], data["calibration"][1])
        artifact = {"id": uid("classifier"), "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
                    "model": spec, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                    "calibration": {"coef": float(cal.coef_[0, 0]), "intercept": float(cal.intercept_[0])},
                    "seed": seed, "dataset_sha256": digest(canonical(rows)), "split_counts": counts, "threshold": .5,
                    "synthetic": all(r.get("synthetic", False) for r in rows)}
        val_rows = [r["features"] for r in rows if r["split"] == "validation"]
        probabilities = predict_artifact(artifact, val_rows)
        # Sweep tied thresholds in descending order, once. Never tune on test data.
        candidates = [(0., 0, 1.000001)]
        order = np.argsort(-probabilities, kind="stable")
        sorted_p, sorted_y = probabilities[order], data["validation"][1][order]
        positives = np.cumsum(sorted_y)
        ends = np.flatnonzero(np.r_[sorted_p[:-1] != sorted_p[1:], True])
        for i in ends:
            tp, fp = int(positives[i]), int(i + 1 - positives[i])
            if tp / (i + 1) >= target_precision:
                candidates.append((tp / positives[-1], -fp, float(sorted_p[i])))
        artifact["threshold"] = max(candidates)[2]
        metrics = binary_metrics(data["validation"][1], probabilities, artifact["threshold"])
        comparisons.append({"model": name, "threshold": artifact["threshold"], "validation": metrics})
        artifacts[name] = artifact
    selected = max(comparisons, key=lambda c: (c["validation"]["f1"], c["validation"]["pr_auc"], -c["validation"]["brier"]))["model"]
    artifact = artifacts[selected]
    test_probs = predict_artifact(artifact, [r["features"] for r in rows if r["split"] == "test"])
    test = binary_metrics(data["test"][1], test_probs, artifact["threshold"])
    artifact["test_metrics"] = test
    failures = [{"identity_a": r["identity_a"], "identity_b": r["identity_b"], "label": r["label"], "probability": float(p)}
                for r, p in zip([r for r in rows if r["split"] == "test"], test_probs) if int(p >= artifact["threshold"]) != r["label"]]
    return artifact, {"selected_model": selected, "comparisons": comparisons, "held_out_test": test,
                      "failures": failures, "split_counts": counts, "synthetic": artifact["synthetic"],
                      "note": "Model selection and threshold use validation only; calibration has its own disjoint split. Test evaluated once after selection."}


def synthetic_dataset(seed=42, pairs_per_split=60):
    rng, rows = np.random.default_rng(seed), []
    for split in SPLITS:
        for i in range(pairs_per_split):
            label = i % 2
            identity = f"{split}_subject_{i}"
            similarity = float(np.clip(rng.normal(.72 if label else .35, .2), -1, 1))
            vals = {"similarity": similarity, "fused_similarity": similarity, "margin": float(rng.uniform(-.2, .4)),
                    "quality_min": float(rng.uniform(.15, .95)), "consistency_min": float(rng.uniform(.5, 1)),
                    "mutual_nn": int(rng.random() < (.8 if label else .3)), "time_known": 0, "time_decay": 0,
                    "route_known": 0, "route_plausible": 0}
            rows.append({"split": split, "label": label, "identity_a": identity, "identity_b": identity if label else identity + "_other",
                         "sequence_a": f"{split}_video_a_{i}", "sequence_b": f"{split}_video_b_{i}", "case_id": f"{split}_case",
                         "feature_version": FEATURE_VERSION, "features": vals, "synthetic": True})
    return rows


def retrieval_metrics(queries: list[dict], k=5):
    """Queries contain ranked identities and a set/list of relevant gallery identities."""
    if not queries or k < 1:
        raise ValueError("Provide queries and a positive K.")
    ranks, aps, recalls, precisions = [], [], [], []
    unknown_false = unknown_n = 0
    for q in queries:
        relevant, ranked = set(q["relevant"]), q["ranked"]
        if not relevant:
            unknown_n += 1
            unknown_false += bool(ranked)
            continue
        hit, ap, found = 0, 0.0, set()
        for i, item in enumerate(ranked, 1):
            if item in relevant and item not in found:
                hit += 1
                ap += hit / i
                found.add(item)
        ranks.append(int(bool(ranked) and ranked[0] in relevant))
        aps.append(ap / len(relevant))
        recalls.append(len(set(ranked[:k]) & relevant) / len(relevant))
        precisions.append(len(set(ranked[:k]) & relevant) / k)
    return {"rank_1": float(np.mean(ranks)) if ranks else None, "mAP": float(np.mean(aps)) if aps else None,
            f"recall_at_{k}": float(np.mean(recalls)) if recalls else None,
            f"precision_at_{k}": float(np.mean(precisions)) if precisions else None,
            "unknown_false_accept_rate": unknown_false / unknown_n if unknown_n else None}
