"""Extensions on the LMB knee-point data:
  (1) inter-cell learning (BatLiNet idea, Zhang et al., Nat. Mach. Intell. 2025)
  (2) how many early cycles are needed (early-prediction horizon, Attia/Severson idea)
  (3) uncertainty-guided choice of which cells to cycle to the knee first (closed-loop idea, Attia et al. 2020)

Outputs fig/kp_fig9_intercell.png, kp_fig10_early.png, kp_fig11_active.png, out/kp_ext.json
"""
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
import kp_data as K
import kp_models as M
from kp_common import save, OUT, BLUE, RED, GREEN, NAVY, ORANGE, GREY, CHEM_COLOR, PAPER_TEST_MAPE

warnings.filterwarnings("ignore")
CFG = dict(batch=8, epochs=500, val_frac=0.1)
ENS = 3
D = K.build()
X, kp, chem = D["X"], D["KP"], D["chem"]
y = M.to_y(kp)
n = len(kp)
out = {}

# ---------------------------------------------------------------- (1) inter-cell
splits = {"random": M.random_splits(n, 10, seed=42), "chem": M.chem_splits(chem)}
ic = {s: {"intra": [], "inter": [], "blend": []} for s in splits}
for sname, sp in splits.items():
    for r, (tr, te) in enumerate(sp):
        a, lo, hi = K.scale_pm1(X[tr]); b = K.scale_pm1(X[te], lo, hi)[0]
        pi = np.mean([M.cnn_predict(M.cnn_train(a, y[tr], seed=50 + s, **CFG), b) for s in range(ENS)], 0)
        pc = np.mean([M.inter_predict(M.cnn_train_inter(a, y[tr], seed=s), b, a, y[tr]) for s in range(ENS)], 0)
        for k_, v in [("intra", pi), ("inter", pc), ("blend", 0.5 * (pi + pc))]:
            ic[sname][k_].append(M.mape(kp[te], M.to_kp(v)))
        print("intercell", sname, r, {k: round(v[-1], 1) for k, v in ic[sname].items()}, flush=True)
out["intercell"] = {s: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "all": [float(x) for x in v]} for k, v in d.items()}
                    for s, d in ic.items()}

fig, ax = plt.subplots(figsize=(10, 4.4))
groups = [("random 80/20\n(10 splits)", [np.mean(ic["random"][k]) for k in ["intra", "inter", "blend"]]),
          ("train NMC → test S", [ic["chem"][k][0] for k in ["intra", "inter", "blend"]]),
          ("train S → test NMC", [ic["chem"][k][1] for k in ["intra", "inter", "blend"]])]
x = np.arange(len(groups))
for k_, (lab, col) in enumerate(zip(["CNN intra-cell (paper)", "CNN inter-cell", "Blend 0.5 / 0.5"], [RED, BLUE, NAVY])):
    v = [g[1][k_] for g in groups]
    ax.bar(x + (k_ - 1) * 0.26, v, 0.26, color=col, label=lab)
    for xi, vi in zip(x + (k_ - 1) * 0.26, v):
        ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("KP MAPE (%)"); ax.legend(fontsize=9, ncol=3, loc="upper left")
ax.set_title("Learning from differences between cells")
fig.tight_layout(); save(fig, "kp_fig9_intercell.png")

# ---------------------------------------------------------------- (2) early-cycle horizon
N_LIST = [2, 3, 4, 6, 8, 10]
sp = M.random_splits(n, 10, seed=42)
eh = {m: {nc: [] for nc in N_LIST} for m in ["Elastic net (seg)", "AV100 per chemistry", "1D CNN"]}
for nc in N_LIST:
    F = M.seg_features(X, n_cycle=nc)
    av = M.area_variation(X, n_cycle=nc)[:, -1]
    for r, (tr, te) in enumerate(sp):
        sc = StandardScaler().fit(F[tr]); ym = y[tr].mean()
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=100000).fit(sc.transform(F[tr]), y[tr] - ym)
        eh["Elastic net (seg)"][nc].append(M.mape(kp[te], M.to_kp(en.predict(sc.transform(F[te])) + ym)))
        pr = np.zeros(len(te))
        for c in np.unique(chem[te]):
            a_, b_ = tr[chem[tr] == c], te[chem[te] == c]
            lr = LinearRegression().fit(np.log10(av[a_, None] + 1e-6), np.log10(kp[a_]))
            pr[chem[te] == c] = 10 ** lr.predict(np.log10(av[b_, None] + 1e-6))
        eh["AV100 per chemistry"][nc].append(M.mape(kp[te], pr))
        if r < 5:
            a, lo, hi = K.scale_pm1(X[tr][:, :, :nc]); b = K.scale_pm1(X[te][:, :, :nc], lo, hi)[0]
            pc = np.mean([M.cnn_predict(M.cnn_train(a, y[tr] * 0 + M.to_y(kp[tr]), seed=s, **CFG), b) for s in range(2)], 0)
            eh["1D CNN"][nc].append(M.mape(kp[te], M.to_kp(pc)))
    print("early", nc, {m: round(np.mean(v[nc]), 1) for m, v in eh.items()}, flush=True)
base = np.mean([M.mape(kp[te], np.array([10 / y[tr][chem[tr] == c].mean() for c in chem[te]])) for tr, te in sp])
out["early"] = {m: {str(nc): float(np.mean(v[nc])) for nc in N_LIST} for m, v in eh.items()}
out["early"]["chem_mean_baseline"] = float(base)
out["early"]["median_kp"] = float(np.median(kp))

fig, ax = plt.subplots(figsize=(9, 4.6))
for m, col in [("Elastic net (seg)", NAVY), ("AV100 per chemistry", GREEN), ("1D CNN", RED)]:
    ax.plot(N_LIST, [np.mean(eh[m][nc]) for nc in N_LIST], "-o", color=col, label=m)
ax.axhline(base, color=GREY, ls="--", lw=1); ax.text(N_LIST[-1], base + 1, "chemistry-mean baseline", ha="right", color=GREY, fontsize=9)
ax.set_xlabel("Early cycles used as input"); ax.set_ylabel("KP MAPE (%), 10 random splits")
sec = ax.secondary_xaxis("top", functions=(lambda c: c / np.median(kp) * 100, lambda p: p * np.median(kp) / 100))
sec.set_xlabel("…as % of the median knee point")
ax.legend(fontsize=9); ax.set_title("How early can the knee be called?", pad=34)
fig.tight_layout(); save(fig, "kp_fig10_early.png")

# ---------------------------------------------------------------- (3) which cells to cycle first
F = M.seg_features(X)
SIZES = list(range(8, 64, 4)) + [63]
al = {"random": [], "uncertainty": []}
for r, (tr, te) in enumerate(M.random_splits(n, 10, seed=7)):
    for strat in al:
        rng = np.random.default_rng(r)
        lab = list(rng.choice(tr, 8, replace=False))
        curve = []
        while True:
            sc = StandardScaler().fit(F[lab])
            g = GaussianProcessRegressor(Ck(1.0) * RBF(np.ones(1) * 3.0, (0.1, 1e3)) + WhiteKernel(0.1), normalize_y=True,
                                         random_state=0).fit(sc.transform(F[lab]), y[lab])
            if len(lab) in SIZES:
                curve.append(M.mape(kp[te], M.to_kp(g.predict(sc.transform(F[te])))))
            pool = np.setdiff1d(tr, lab)
            if len(pool) == 0 or len(lab) >= SIZES[-1]:
                break
            if strat == "uncertainty":
                _, s = g.predict(sc.transform(F[pool]), return_std=True)
                lab.append(int(pool[np.argmax(s)]))
            else:
                lab.append(int(rng.choice(pool)))
        al[strat].append(curve)
    print("active", r, flush=True)
L = min(len(c) for v in al.values() for c in v)
out["active"] = {"sizes": SIZES[:L], **{k: [float(x) for x in np.mean([c[:L] for c in v], 0)] for k, v in al.items()}}
fig, ax = plt.subplots(figsize=(9, 4.4))
for k_, col in [("random", GREY), ("uncertainty", BLUE)]:
    A = np.array([c[:L] for c in al[k_]])
    ax.fill_between(SIZES[:L], A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=col, alpha=0.18, lw=0)
    ax.plot(SIZES[:L], A.mean(0), "-o", ms=4, color=col, label={"random": "random cells", "uncertainty": "most uncertain cell first"}[k_])
ax.set_xlabel("Cells cycled until the knee (labelled)"); ax.set_ylabel("Test KP MAPE (%)")
ax.legend(fontsize=9); ax.set_title("Which cells should be cycled to the knee first? (GPR on segment features)")
fig.tight_layout(); save(fig, "kp_fig11_active.png")

(OUT / "kp_ext.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "intercell"}, indent=1))
print({s: {k: round(v["mean"], 1) for k, v in d.items()} for s, d in out["intercell"].items()})
