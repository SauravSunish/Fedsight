"""Opt-in numerical research baselines; not a deployed federation or privacy claim."""
from __future__ import annotations
import numpy as np


def federated_simulation(clients, *, rounds=10, local_steps=5, learning_rate=.1, mu=.01, seed=42):
    """FedAvg (mu=0) / FedProx on a logistic pair classifier, in one process."""
    if not 2 <= len(clients) <= 20 or not 1 <= rounds <= 200 or not 1 <= local_steps <= 100:
        raise ValueError("Use 2–20 clients, 1–200 rounds and 1–100 local steps.")
    if not np.isfinite([learning_rate, mu]).all() or not 0 < learning_rate <= 1 or mu < 0:
        raise ValueError("Invalid optimiser configuration.")
    validated = []
    dimension = None
    for x, y in clients:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Each client requires finite features and binary labels.")
        if dimension is not None and x.shape[1] != dimension:
            raise ValueError("Client feature dimensions must agree.")
        dimension = x.shape[1]
        validated.append((np.column_stack([x, np.ones(len(x))]), y))
    global_weights = np.zeros(dimension + 1)
    history = []
    for round_id in range(rounds):
        updates, sizes = [], []
        for x, y in validated:
            local = global_weights.copy()
            for _ in range(local_steps):
                prediction = 1 / (1 + np.exp(-np.clip(x @ local, -60, 60)))
                gradient = x.T @ (prediction - y) / len(y) + mu * (local - global_weights)
                local -= learning_rate * gradient
            updates.append(local)
            sizes.append(len(y))
        global_weights = np.average(updates, axis=0, weights=sizes)
        losses = []
        for x, y in validated:
            logits = x @ global_weights
            losses.append(float(np.mean(np.logaddexp(0, logits) - y * logits)))
        history.append({"round": round_id + 1, "weighted_training_loss": float(np.average(losses, weights=sizes))})
    return {"method": "FedProx" if mu else "FedAvg", "weights": global_weights.tolist(), "rounds": history,
            "seed": seed, "client_samples": sizes, "privacy": "Local simulation only. No secure aggregation, differential privacy or network federation."}
