"""Benchmark of model families on the LMB knee-point task + DoD-segment attribution consensus.

Three validation schemes:
  random  : 20 random 80/20 splits (the paper used one such split)
  batch   : GroupKFold(5) over assembly batches (cells made the same day never straddle train/test)
  chem    : train on one chemistry, test on the other (NMC -> S, S -> NMC)

Outputs fig/kp_fig6_benchmark.png, kp_fig7_consensus.png, kp_fig8_uncertainty.png, out/kp_bench.json
"""
import json
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
import kp_data as K
import kp_models as M
from kp_common import save, OUT, BLUE, RED, GREEN, NAVY, ORANGE, GREY, PURPLE, CHEM_COLOR, CHEM_LABEL, PAPER_TEST_MAPE

sys.path.insert(0, "/home/claude/eis/src")
from gp_jax import fit_ard_restarts  # noqa: E402  (same JAX ARD-GP as the EIS study)

warnings.filterwarnings("ignore")
CNN_CFG = dict(batch=8, epochs=500, val_frac=0.1)
CNN_ENS = 3

D = K.build()
X, kp, chem, batch = D["X"], D["KP"], D["chem"], D["batch"]
y = M.to_y(kp)
n = len(kp)
SEG = M.seg_index()
F_seg = M.seg_features(X)
AV = M.area_variation(X)


def fit_predict(name, tr, te, want_std=False):
    """Returns predicted y on te (and optional std, and per-feature info)."""
    info = {}
    if name == "Overall mean":
        return np.full(len(te), y[tr].mean()), None, info
    if name == "Chemistry mean":
        return np.array([y[tr][chem[tr] == c].mean() for c in chem[te]]), None, info
    if name == "AV100 per chemistry":
        out = np.zeros(len(te))
        for c in np.unique(chem[te]):
            a, b = tr[chem[tr] == c], te[chem[te] == c]
            src = a if len(a) >= 3 else tr                       # chem-transfer: fall back to pooled fit
            lr = LinearRegression().fit(np.log10(AV[src, -1:]), np.log10(kp[src]))
            out[chem[te] == c] = M.to_y(10 ** lr.predict(np.log10(AV[b, -1:])))
        return out, None, info
    if name.endswith("(seg)"):
        sc = StandardScaler().fit(F_seg[tr]); A, B = sc.transform(F_seg[tr]), sc.transform(F_seg[te])
    else:
        Xs, lo, hi = K.scale_pm1(X[tr]); Xt = K.scale_pm1(X[te], lo, hi)[0]
        A, B = M.raw_features(Xs), M.raw_features(Xt)
    ym = y[tr].mean()
    if name.startswith("Elastic net"):
        fast = A.shape[1] > 100
        m = ElasticNetCV(l1_ratio=[0.5] if fast else [0.1, 0.5, 0.9], cv=5, alphas=20 if fast else 100,
                         max_iter=2000 if fast else 100000, tol=1e-3 if fast else 1e-4).fit(A, y[tr] - ym)
        info["coef"] = m.coef_
        return m.predict(B) + ym, None, info
    if name.startswith("GPR-ARD"):
        best, _ = fit_ard_restarts(A, y[tr] - ym, n_restarts=3, seed=int(te[0]))
        mu, sd = best.predict(B, return_std=True)
        info["ls"] = best.length_scales
        return mu + ym, sd, info
    if name.startswith("GPR-RBF"):
        k = Ck(1.0) * RBF(10.0, (1e-1, 1e4)) + WhiteKernel(0.1)
        m = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(A, y[tr])
        mu, sd = m.predict(B, return_std=True)
        return mu, sd, info
    if name.startswith("KRR"):
        m = KernelRidge(alpha=0.1, kernel="rbf", gamma=1.0 / A.shape[1]).fit(A, y[tr] - ym)
        return m.predict(B) + ym, None, info
    if name.startswith("SVR"):
        m = SVR(C=1.0, epsilon=0.02, gamma="scale").fit(A, y[tr])
        return m.predict(B), None, info
    if name.startswith("k-NN"):
        m = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(A, y[tr])
        return m.predict(B), None, info
    if name.startswith("Random forest"):
        m = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(A, y[tr])
        info["imp"] = m.feature_importances_
        return m.predict(B), None, info
    if name.startswith("Gradient boosting"):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=5, random_state=0).fit(A, y[tr])
        return m.predict(B), None, info
    if name.startswith("MLP"):
        m = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=5000, random_state=0).fit(A, y[tr] - ym)
        return m.predict(B) + ym, None, info
    if name.startswith("1D CNN"):
        Xs, lo, hi = K.scale_pm1(X[tr]); Xt = K.scale_pm1(X[te], lo, hi)[0]
        P, perm = [], []
        for s in range(CNN_ENS):
            p = M.cnn_train(Xs, y[tr], seed=7 * int(te[0]) + s, **CNN_CFG)
            P.append(M.cnn_predict(p, Xt))
            if want_std:                      # permutation importance per DoD segment (test cells)
                rng = np.random.default_rng(s)
                base = M.mape(kp[te], M.to_kp(P[-1]))
                row = []
                for sidx in SEG:
                    Xp = Xt.copy(); Xp[:, sidx, :] = Xt[rng.permutation(len(te))][:, sidx, :]
                    row.append(M.mape(kp[te], M.to_kp(M.cnn_predict(p, Xp))) - base)
                perm.append(row)
        if perm:
            info["perm"] = np.mean(perm, 0)
        return np.mean(P, 0), None, info
    raise ValueError(name)


MODELS = ["Overall mean", "Chemistry mean", "AV100 per chemistry", "Elastic net (seg)", "Elastic net (raw)",
          "GPR-ARD (seg)", "GPR-RBF (raw)", "KRR (raw)", "SVR (raw)", "k-NN (raw)", "Random forest (seg)",
          "Random forest (raw)", "Gradient boosting (seg)", "MLP (raw)", "1D CNN (paper, raw)"]
SCHEMES = {"random": M.random_splits(n, 20, seed=42), "batch": M.group_splits(batch, 5), "chem": M.chem_splits(chem)}

results = {s: {m: [] for m in MODELS} for s in SCHEMES}
imp = {"Elastic net |coef|": [], "Random forest impurity": [], "GPR-ARD 1/length-scale": [], "1D CNN permutation": [],
       "Elastic net permutation": []}
cover = {"GPR-ARD (seg)": [], "GPR-RBF (raw)": []}
for sname, splits in SCHEMES.items():
    for r, (tr, te) in enumerate(splits):
        for mname in MODELS:
            if mname.startswith("1D CNN") and sname == "random" and r >= 10:
                continue                                         # 10 CNN repeats here (20 in kp_repro.py)
            yh, sd, info = fit_predict(mname, tr, te, want_std=(sname == "random"))
            kh = M.to_kp(yh)
            results[sname][mname].append({"mape": M.mape(kp[te], kh), "mae": M.mae(kp[te], kh),
                                          "mape_nmc": M.mape(kp[te][chem[te] == "NMC"], kh[chem[te] == "NMC"]) if (chem[te] == "NMC").any() else None,
                                          "mape_s": M.mape(kp[te][chem[te] == "S"], kh[chem[te] == "S"]) if (chem[te] == "S").any() else None})
            if sname == "random":
                seg_of = M.SEG_FEATURE_SEGMENT
                agg = lambda v: np.array([np.abs(v)[seg_of == s].sum() for s in range(10)])
                if "coef" in info and mname == "Elastic net (seg)":
                    imp["Elastic net |coef|"].append(agg(info["coef"]))
                    # permutation per segment for the elastic net (refit-free)
                    sc = StandardScaler().fit(F_seg[tr])
                    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=100000).fit(sc.transform(F_seg[tr]), y[tr] - y[tr].mean())
                    B = sc.transform(F_seg[te]); base = M.mape(kp[te], M.to_kp(en.predict(B) + y[tr].mean()))
                    rng = np.random.default_rng(r); row = []
                    for s in range(10):
                        Bp = B.copy(); cols = np.where(seg_of == s)[0]; Bp[:, cols] = B[rng.permutation(len(te))][:, cols]
                        row.append(M.mape(kp[te], M.to_kp(en.predict(Bp) + y[tr].mean())) - base)
                    imp["Elastic net permutation"].append(np.array(row))
                if "imp" in info and mname == "Random forest (seg)":
                    imp["Random forest impurity"].append(agg(info["imp"]))
                if "ls" in info:
                    imp["GPR-ARD 1/length-scale"].append(agg(1.0 / info["ls"]))
                if "perm" in info:
                    imp["1D CNN permutation"].append(np.array(info["perm"]))
                if sd is not None:
                    z = np.abs(yh - y[te]) / np.maximum(sd, 1e-9)
                    cover[mname].append([float((z <= 1).mean()), float((z <= 2).mean())])
            print(sname, r, mname, round(results[sname][mname][-1]["mape"], 1), flush=True)

summary = {s: {m: {"mape_mean": float(np.mean([v["mape"] for v in results[s][m]])),
                   "mape_sd": float(np.std([v["mape"] for v in results[s][m]])),
                   "mae_mean": float(np.mean([v["mae"] for v in results[s][m]]))} for m in MODELS} for s in SCHEMES}
summary["chem_detail"] = {m: {"NMC->S": results["chem"][m][0]["mape"], "S->NMC": results["chem"][m][1]["mape"]} for m in MODELS}

# ---------------------------------------------------------------- Fig 6: benchmark
order = sorted(MODELS, key=lambda m: summary["random"][m]["mape_mean"])
fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
yy = np.arange(len(order))
col = [GREY if "mean" in m else (RED if "CNN" in m else (GREEN if "AV100" in m else NAVY)) for m in order]
ax[0].barh(yy, [summary["random"][m]["mape_mean"] for m in order], xerr=[summary["random"][m]["mape_sd"] for m in order],
           color=col, alpha=0.85, error_kw=dict(lw=0.8, color=GREY))
ax[0].set_yticks(yy); ax[0].set_yticklabels(order, fontsize=10); ax[0].invert_yaxis()
ax[0].axvline(PAPER_TEST_MAPE, color=RED, ls="--", lw=1); ax[0].text(PAPER_TEST_MAPE + 0.5, len(order) - 0.6, "paper 15.2 %", color=RED, fontsize=9)
ax[0].set_xlabel("KP MAPE (%) — 20 random 80/20 splits, mean ± sd")
ax[0].set_title("a   Same 79 cells, same target (10/KP), different model families")
Mx = np.array([[summary["random"][m]["mape_mean"], summary["batch"][m]["mape_mean"],
                summary["chem_detail"][m]["NMC->S"], summary["chem_detail"][m]["S->NMC"]] for m in order])
im = ax[1].imshow(np.clip(Mx, 0, 120), cmap="Blues", aspect="auto", vmin=10, vmax=120)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["random\n80/20", "new batch\n(GroupKFold)", "train NMC\ntest S", "train S\ntest NMC"], fontsize=9.5)
ax[1].set_yticks(yy); ax[1].set_yticklabels(order, fontsize=10)
for i in range(Mx.shape[0]):
    for j in range(Mx.shape[1]):
        ax[1].text(j, i, f"{Mx[i, j]:.0f}", ha="center", va="center", fontsize=9, color="w" if Mx[i, j] > 75 else "k")
ax[1].set_title("b   KP MAPE (%) under harder validation")
plt.colorbar(im, ax=ax[1], fraction=0.03)
fig.tight_layout(); save(fig, "kp_fig6_benchmark.png")

# ---------------------------------------------------------------- Fig 7: attribution consensus over DoD segments
cnn = np.load(OUT / "cnn_oos.npz")
cam = cnn["cam"]; has = cnn["has"]
rows = {}
rows["1D CNN Grad-CAM (NMC)"] = np.array([cam[has & (chem == "NMC")][:, s].mean() for s in SEG.__iter__()]) if False else \
    np.array([cam[has & (chem == "NMC")][:, sidx].mean() for sidx in SEG])
rows["1D CNN Grad-CAM (S)"] = np.array([cam[has & (chem == "S")][:, sidx].mean() for sidx in SEG])
for k_, v in imp.items():
    if v:
        rows[k_] = np.clip(np.mean(v, 0), 0, None)
for c in ["NMC", "S"]:
    mm = chem == c
    rows[f"|Spearman ρ(AV_s, KP)| ({c})"] = np.array([abs(spearmanr(AV[mm, s], kp[mm])[0]) for s in range(10)])
names_ = list(rows)
R = np.array([rows[k] / max(rows[k].sum(), 1e-12) for k in names_])
fig, ax = plt.subplots(figsize=(13, 0.55 * len(names_) + 1.6))
im = ax.imshow(R / R.max(1, keepdims=True), cmap="magma_r", aspect="auto", vmin=0, vmax=1)
ax.set_yticks(range(len(names_))); ax.set_yticklabels(names_, fontsize=10)
ax.set_xticks(range(10)); ax.set_xticklabels([f"{10 * s}–{10 * (s + 1)}%" for s in range(10)], fontsize=9.5)
ax.set_xlabel("DoD segment of the discharge curve (each row scaled to its own max)")
for i in range(R.shape[0]):
    top = np.argmax(R[i]); ax.text(top, i, "★", ha="center", va="center", color="w", fontsize=12)
ax.set_title("Which part of the discharge curve carries the knee-point signal? (★ = row maximum)")
plt.colorbar(im, ax=ax, fraction=0.025)
fig.tight_layout(); save(fig, "kp_fig7_consensus.png")
summary["segment_relevance"] = {k: [float(v) for v in R[i]] for i, k in enumerate(names_)}
summary["last10_share"] = {k: float(R[i, -1]) for i, k in enumerate(names_)}
summary["top_segment"] = {k: int(np.argmax(R[i])) for i, k in enumerate(names_)}

# ---------------------------------------------------------------- Fig 8: is the uncertainty honest?
fig, ax = plt.subplots(figsize=(6.5, 4.4))
summary["coverage"] = {}
for k_, (m, v) in enumerate(cover.items()):
    v = np.array(v); c1, c2 = v[:, 0].mean(), v[:, 1].mean()
    summary["coverage"][m] = {"within_1sd": float(c1), "within_2sd": float(c2)}
    ax.bar(np.array([0, 1]) + (k_ - 0.5) * 0.35, [c1, c2], 0.35, color=[NAVY, BLUE][k_], label=m)
ax.plot([-0.4, 0.4], [0.683, 0.683], "--", color=RED); ax.plot([0.6, 1.4], [0.954, 0.954], "--", color=RED)
ax.text(0, 0.70, "ideal 68 %", ha="center", color=RED, fontsize=9); ax.text(1, 0.97, "ideal 95 %", ha="center", color=RED, fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["|error| ≤ 1σ", "|error| ≤ 2σ"]); ax.set_ylim(0, 1.08)
ax.set_ylabel("Share of test cells"); ax.legend(fontsize=9, loc="lower right")
ax.set_title("GPR predictive intervals on held-out cells")
fig.tight_layout(); save(fig, "kp_fig8_uncertainty.png")

(OUT / "kp_bench.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({s: {m: round(v["mape_mean"], 1) for m, v in summary[s].items()} for s in SCHEMES}, indent=1))
print(json.dumps(summary["chem_detail"], indent=1)); print(summary["top_segment"]); print(summary["coverage"])
