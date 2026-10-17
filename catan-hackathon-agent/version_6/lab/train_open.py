"""Fit the logistic opening model on opengen data. Usage: python train_open.py data.npz [data2.npz ...]
Writes open_logreg.json (standardisation + coefficients). Reports held-out accuracy and log-loss."""
import json
import sys

import numpy as np

from openfeat import NAMES

X = np.concatenate([np.load(p)["X"] for p in sys.argv[1:]]).astype(np.float64)
y = np.concatenate([np.load(p)["y"] for p in sys.argv[1:]]).astype(np.float64)
n = len(y)
cut = int(n * 0.85)
mean, std = X[:cut].mean(0), X[:cut].std(0) + 1e-6
Z = (X - mean) / std
w, b, lam, lr = np.zeros(Z.shape[1]), 0.0, 1e-3, 0.5
for it in range(3000):
    p = 1 / (1 + np.exp(-(Z[:cut] @ w + b)))
    g = p - y[:cut]
    w -= lr * (Z[:cut].T @ g / cut + lam * w)
    b -= lr * g.mean()
pv = 1 / (1 + np.exp(-(Z[cut:] @ w + b)))
acc = ((pv > 0.5) == y[cut:]).mean()
ll = -np.mean(y[cut:] * np.log(pv + 1e-9) + (1 - y[cut:]) * np.log(1 - pv + 1e-9))
print(f"n={n} train={cut} held-out acc={acc:.3f} logloss={ll:.4f} (base rate {y.mean():.3f})")
names = [f"me_{s}" for s in NAMES] + [f"opp_{s}" for s in NAMES] + ["i_first"]
for i in np.argsort(-np.abs(w))[:12]:
    print(f"  {names[i]:22s} {w[i]:+.3f}")
json.dump(dict(mean=mean.tolist(), std=std.tolist(), coef=w.tolist(), bias=float(b)),
          open("open_logreg.json", "w"))
