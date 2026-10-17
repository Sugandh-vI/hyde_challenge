"""Retrain the value-net ensemble (Step 7) on the shards `gen_data.py` writes.

    python version_6/tools/train_net.py --data version_6/data --out version_6/nets

* Split by **game seed** (`--val-mod 8`: every 8th seed is validation), never by row, so the two
  seats of one game never straddle the split.
* Same architecture and file format as the nets in `version_6/base` (102 -> 128 -> 64 -> 1, ReLU, one
  logit; `npz` with `mu, sd, nlayers, W{k}, b{k}`), so `v4net.Value` loads them unchanged.
* Reports validation log-loss / Brier / accuracy of the *current* ensemble on the same split, so
  "better" means better than what we ship, not better than chance.
* Three models with different torch seeds, exactly like the shipped ensemble.

Keep the new nets only if validation log-loss improves **and** the paired screen against `base` comes
out KEEP; otherwise the old weights stay.
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover - torch is a hard requirement for training
    HAS_TORCH = False


def load_shards(pattern):
    X, y, s = [], [], []
    for path in sorted(glob.glob(pattern)):
        d = np.load(path)
        X.append(d["X"])
        y.append(d["y"])
        s.append(d["seed"])
    if not X:
        raise SystemExit(f"no shards matched {pattern}")
    return (np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.float32),
            np.concatenate(s).astype(np.int64))


def forward_np(x, mu, sd, layers):
    h = (x - mu) / sd
    last = len(layers) - 1
    for k, (W, b) in enumerate(layers):
        h = h @ W + b
        if k < last:
            np.maximum(h, 0.0, out=h)
    return h[:, 0]


def baseline_logloss(paths, X, y):
    """Log-loss / Brier / accuracy of an existing ensemble (mean of logits, then sigmoid)."""
    z = 0.0
    for p in paths:
        d = np.load(p)
        n = int(d["nlayers"])
        z = z + forward_np(X.astype(np.float64), d["mu"], d["sd"],
                           [(d[f"W{k}"], d[f"b{k}"]) for k in range(n)])
    z = z / len(paths)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return ll, float(np.mean((p - y) ** 2)), float(np.mean((p > 0.5) == (y > 0.5)))


def train_one(Xtr, ytr, Xva, yva, seed, epochs, batch, lr, hidden, wd, verbose=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden[0]), nn.ReLU(),
                        nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                        nn.Linear(hidden[1], 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.BCEWithLogitsLoss()
    xt = Xtr if torch.is_tensor(Xtr) else torch.from_numpy(Xtr)
    xv = Xva if torch.is_tensor(Xva) else torch.from_numpy(Xva)
    yt = torch.from_numpy(np.asarray(ytr, dtype=np.float32)).unsqueeze(1)
    yv = torch.from_numpy(np.asarray(yva, dtype=np.float32)).unsqueeze(1)
    n = len(xt)
    best = (float("inf"), None)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = torch.from_numpy(order[i:i + batch].astype(np.int64))
            opt.zero_grad()
            out = loss(net(xt[idx]), yt[idx])
            out.backward()
            opt.step()
        with torch.no_grad():
            vl = float(loss(net(xv), yv))
        if vl < best[0]:
            best = (vl, [p.detach().clone() for p in net.parameters()])
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}: val logloss {vl:.4f} (best {best[0]:.4f})", flush=True)
    return best


def export(path, state, mu, sd, hidden):
    W0, b0, W1, b1, W2, b2 = [p.numpy() for p in state]
    np.savez(path, mu=mu, sd=sd, nlayers=np.int64(3),
             W0=W0.T.copy(), b0=b0, W1=W1.T.copy(), b1=b1, W2=W2.T.copy(), b2=b2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="version_6/data/shard_*.npz")
    ap.add_argument("--out", default="version_6/nets")
    ap.add_argument("--val-mod", type=int, default=8)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs=2, default=(128, 64))
    ap.add_argument("--baseline", default="version_6/base/value_w*.npz")
    args = ap.parse_args()

    X, y, seed = load_shards(args.data)
    if len(X) == 0:
        raise SystemExit("empty data")
    val = (seed % args.val_mod) == 0
    print(f"rows {len(X)} (train {int((~val).sum())}, val {int(val.sum())}), "
          f"val seeds {len(np.unique(seed[val]))} of {len(np.unique(seed))}, "
          f"win rate {y.mean():.3f}")
    mu = X[~val].mean(axis=0)
    sd = X[~val].std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xtr = torch_ready(X[~val], mu, sd) if HAS_TORCH else ((X[~val] - mu) / sd)
    Xva = torch_ready(X[val], mu, sd) if HAS_TORCH else ((X[val] - mu) / sd)

    base = sorted(glob.glob(args.baseline))
    if base:
        ll, br, ac = baseline_logloss(base, X[val], y[val])
        print(f"current ensemble  : val logloss {ll:.4f}  brier {br:.4f}  acc {ac:.4f}")

    os.makedirs(args.out, exist_ok=True)
    lls = []
    for k in range(args.models):
        best_ll, state = train_one(Xtr, y[~val], Xva, y[val], 1234 + k, args.epochs, args.batch,
                                   args.lr, tuple(args.hidden), args.wd)
        export(os.path.join(args.out, f"value_w{k}.npz"), state, mu, sd, args.hidden)
        lls.append(best_ll)
        print(f"model {k}: best val logloss {best_ll:.4f}", flush=True)

    # the ensemble as the agent uses it: mean of logits
    import torch
    z = 0.0
    for k in range(args.models):
        d = np.load(os.path.join(args.out, f"value_w{k}.npz"))
        z = z + forward_np(X[val].astype(np.float64), d["mu"], d["sd"],
                           [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["nlayers"]))])
    p = np.clip(1.0 / (1.0 + np.exp(-z / args.models)), 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y[val] * np.log(p) + (1 - y[val]) * np.log(1 - p)))
    print(f"new ensemble      : val logloss {ll:.4f}  brier {float(np.mean((p - y[val]) ** 2)):.4f}"
          f"  acc {float(np.mean((p > 0.5) == (y[val] > 0.5))):.4f}")
    print(f"single models     : {['%.4f' % v for v in lls]}")


def torch_ready(X, mu, sd):
    import torch
    return torch.from_numpy(((X - mu) / sd).astype(np.float32))


if __name__ == "__main__":
    main()
