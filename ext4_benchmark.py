"""Extension 4 — model benchmark and feature attribution (Ng et al., Nat. Mach. Intell. 2, 161 (2020) review).

The review compares the ML families used for SoC/SoH/RUL (GPR, SVR, random forest, boosting,
neural networks, linear models) and stresses interpretability. Here every family gets the same
input (one state-V EIS spectrum, 120 features), the same split as the paper's Fig. 3
(train 25C01–04, 35C01, 45C01; test 25C05–08, 35C02, 45C02) and the same target (capacity, mAh).

Then: do different models agree on WHICH frequencies matter?  Permutation importance (model-agnostic),
random-forest impurity importance and elastic-net |coefficients| are compared with the CNN saliency
(ext1) and the paper's ARD weights.

Outputs fig/ext4a_benchmark.png, fig/ext4b_attribution_consensus.png, fig/ext4_metrics.json
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import eis_data as D
import models as M
from common import stack, save, FIG, BLUE, RED, ORANGE, NAVY, GREY, GREEN, TRAIN_MT, TEST_25, TEST_MT

TEST = TEST_25 + TEST_MT
X, y, g, _ = stack(TRAIN_MT)
mu, sd = D.zscore_fit(X)
Z = (X - mu) / sd
ym = y.mean()
tests = {c: D.paired(c) for c in TEST}
FREQ = D.FREQ

zoo = {
    "GPR SE-iso (paper)": M.gpr_se_iso(),
    "SVR (RBF)": SVR(C=10.0, epsilon=0.1, gamma="scale"),
    "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "Gradient boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0),
    "Elastic net": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=50000),
    "k-NN (k=10)": KNeighborsRegressor(n_neighbors=10, weights="distance"),
    "MLP (2×64)": MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=3000, early_stopping=True, random_state=0),
}
center = {"SVR (RBF)", "Elastic net", "MLP (2×64)"}       # these work on a centred target
table = {}
for name, m in zoo.items():
    m.fit(Z, y - ym if name in center else y)
    row = {}
    for c, d in tests.items():
        p = m.predict((d["X"] - mu) / sd) + (ym if name in center else 0)
        row[c] = float(np.sqrt(np.mean((d["cap"] - p) ** 2)))
    row["mean"] = float(np.mean([row[c] for c in TEST]))
    table[name] = row
try:
    e1 = json.loads((FIG / "ext1_metrics.json").read_text())
    table["1D CNN ×5 (ext1)"] = {c: e1[c]["rmse_cnn"] for c in TEST}
    table["1D CNN ×5 (ext1)"]["mean"] = float(np.mean([e1[c]["rmse_cnn"] for c in TEST]))
except Exception:
    pass

names = sorted(table, key=lambda n: table[n]["mean"])
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].barh(names, [table[n]["mean"] for n in names], color=[NAVY if "paper" in n else (ORANGE if "CNN" in n else GREY) for n in names])
ax[0].invert_yaxis(); ax[0].set_xlabel("Mean test RMSE over 6 cells (mAh)")
ax[0].set_title("a   Same input, same split, different model families")
Mx = np.array([[table[n][c] for c in TEST] for n in names])
im = ax[1].imshow(Mx, cmap="Blues", aspect="auto")
ax[1].set_xticks(range(len(TEST))); ax[1].set_xticklabels(TEST); ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names)
for i in range(Mx.shape[0]):
    for j in range(Mx.shape[1]):
        ax[1].text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=9,
                   color="w" if Mx[i, j] > Mx.max() * 0.6 else "k")
ax[1].set_title("b   RMSE per test cell (mAh)")
plt.colorbar(im, ax=ax[1], fraction=0.03)
fig.tight_layout(); save(fig, "ext4a_benchmark.png")

# ---- attribution consensus
Xt = np.vstack([(d["X"] - mu) / sd for d in tests.values()]); yt = np.concatenate([d["cap"] for d in tests.values()])
attr = {}
pi = permutation_importance(zoo["GPR SE-iso (paper)"], Xt, yt, n_repeats=5, random_state=0, scoring="neg_mean_squared_error")
attr["Permutation, GPR (test cells)"] = np.clip(pi.importances_mean, 0, None)
attr["Random-forest impurity"] = zoo["Random forest"].feature_importances_
attr["Elastic-net |coef|"] = np.abs(zoo["Elastic net"].coef_)
try:
    a1 = np.load(FIG / "ext1_arrays.npz")
    attr["CNN saliency (ext1)"] = a1["saliency"].T.ravel() if a1["saliency"].shape == (60, 2) else a1["saliency"]
    if "ard" in a1.files:
        attr["ARD weight (paper method)"] = a1["ard"]
except Exception:
    pass
A = np.array([v / v.sum() for v in attr.values()])

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for j, part in enumerate(["Re(Z)", "−Im(Z)"]):
    sub = A[:, j * 60:(j + 1) * 60]
    sub = sub / sub.max(1, keepdims=True)
    ax[j].imshow(sub, aspect="auto", cmap="magma_r", extent=[np.log10(FREQ[0]), np.log10(FREQ[-1]), len(attr) - 0.5, -0.5])
    ax[j].set_yticks(range(len(attr))); ax[j].set_yticklabels(list(attr), fontsize=9)
    ax[j].set_title(f"{part}: relevance per method (each row scaled to its own max)")
    for f0, ls in [(17.8, "--"), (2.16, ":")]:
        ax[j].axvline(np.log10(f0), color=BLUE, lw=1, ls=ls)
ax[1].set_xlabel("log10 frequency (Hz)   — high frequency on the left")
fig.tight_layout(); save(fig, "ext4b_attribution_consensus.png")

band = lambda v, lo, hi: float(v[[i for i in range(120) if lo <= FREQ[i % 60] < hi]].sum())
bands = [(0.0, 1.0, "< 1 Hz (diffusion tail)"), (1.0, 100.0, "1–100 Hz (charge transfer / SEI)"), (100.0, 1e5, "> 100 Hz (ohmic, contacts)")]
share = {n: {b[2]: round(band(v / v.sum(), b[0], b[1]), 3) for b in bands} for n, v in attr.items()}
out = {"rmse_mAh": {n: {k: round(v, 3) for k, v in r.items()} for n, r in table.items()}, "relevance_share_by_band": share}
(FIG / "ext4_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
