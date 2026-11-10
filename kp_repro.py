"""Reproduce the core of Liu et al. (JACS 2024): KP extraction, 1D CNN on the first 10 discharge curves,
Grad-CAM saliency by KP bin, and the AV descriptor family (AV10 ... AV100).

Outputs fig/kp_fig1_data.png ... kp_fig5_av.png and out/kp_repro.json, out/cnn_oos.npz
"""
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import kp_data as K
import kp_models as M
from kp_common import save, OUT, BLUE, RED, GREEN, NAVY, ORANGE, GREY, CHEM_COLOR, CHEM_LABEL, PAPER_TEST_MAPE, PAPER_TRAIN_MAPE, dod_axis

warnings.filterwarnings("ignore")
CFG = dict(batch=8, epochs=500, val_frac=0.1)   # set from the settings check (see README)
ENS = 3
REPEATS = 20

D = K.build()
X, kp, chem, Q, names = D["X"], D["KP"], D["chem"], D["Q"], D["names"]
y = M.to_y(kp)
n = len(kp)
DOD = D["dod"]
res = {"n_cells": int(n), "n_nmc": int((chem == "NMC").sum()), "n_s": int((chem == "S").sum())}

# ---------------------------------------------------------------- Fig 1: data and KP extraction
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
for j, c in enumerate(["NMC", "S"]):
    a = ax[j]
    for i in np.where(chem == c)[0]:
        q = np.asarray(Q[i], float)
        r = q / q[0]
        a.plot(np.arange(len(q)), r, color=CHEM_COLOR[c], lw=0.7, alpha=0.45)
        k = int(kp[i])
        if k < len(r):
            a.plot(k, r[k], "o", ms=3.5, color=NAVY)
    a.axvspan(0, 9, color=GREEN, alpha=0.12, lw=0)
    a.text(1, 0.62, "cycles 0–9\n(model input)", fontsize=9, color=GREEN)
    a.set_ylim(0.55, 1.12); a.set_xlim(0, 120)
    a.set_xlabel("Cycle number"); a.set_ylabel("Capacity retention")
    a.set_title(f"{'ab'[j]}   {CHEM_LABEL[c]}  (n = {(chem == c).sum()}, dots = KP)")
bins = np.arange(5, 80, 5)
for c in ["NMC", "S"]:
    ax[2].hist(kp[chem == c], bins, histtype="step", lw=2, color=CHEM_COLOR[c], label=f"{c}: this work")
    ax[2].hist(K.TABLE_S1_KP[c], bins, histtype="step", lw=1.2, ls="--", color=CHEM_COLOR[c], label=f"{c}: SI Table S1")
ax[2].set_xlabel("Knee point (cycle)"); ax[2].set_ylabel("Cells"); ax[2].legend(fontsize=9)
ax[2].set_title("c   KP distribution vs paper")
fig.tight_layout(); save(fig, "kp_fig1_data.png")
res["kp_match"] = {}
for c in ["NMC", "S"]:
    a_, b_ = np.sort(kp[chem == c]), np.sort(K.TABLE_S1_KP[c])
    res["kp_match"][c] = {"exact_sorted_matches": int((a_ == b_).sum()), "n": int(len(a_)),
                          "max_sorted_diff": float(np.abs(a_ - b_).max()), "mean_ours": float(a_.mean()), "mean_paper": float(b_.mean())}

# ---------------------------------------------------------------- Fig 2: what the input looks like
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for j, c in enumerate(["S", "NMC"][:1]):
    pass
m = chem == "S"
lo_i = np.where(m)[0][np.argmin(kp[m])]; hi_i = np.where(m)[0][np.argmax(kp[m])]
for i, ls, lab in [(lo_i, "-", "short life"), (hi_i, "--", "long life")]:
    for cyc in range(10):
        ax[0].plot(DOD, X[i, :, cyc], ls, color=plt.cm.viridis(cyc / 9), lw=1)
    ax[0].plot([], [], ls, color=GREY, label=f"{lab}: KP {int(kp[i])}")
dod_axis(ax[0]); ax[0].set_ylabel("U (V)"); ax[0].legend(fontsize=9)
ax[0].set_title("a   Li|S: cycles 0–9 (colour = cycle)")
for j, c in enumerate(["NMC", "S"]):
    a = ax[1 + j]
    mm = chem == c
    t1, t2 = np.percentile(kp[mm], [33.3, 66.7])
    groups = [(kp <= t1) & mm, (kp > t1) & (kp <= t2) & mm, (kp > t2) & mm]
    for g, col, lab in zip(groups, [RED, GREY, GREEN], ["early KP (lower third)", "middle third", "late KP (upper third)"]):
        dv = (X[g, :, 9] - X[g, :, 0]) * 1000
        a.plot(DOD, dv.mean(0), color=col, lw=2, label=lab)
        a.fill_between(DOD, np.percentile(dv, 25, 0), np.percentile(dv, 75, 0), color=col, alpha=0.12, lw=0)
    a.axvspan(0.9, 1.0, color=NAVY, alpha=0.08, lw=0)
    a.axhline(0, color=GREY, lw=0.6)
    dod_axis(a); a.set_ylabel("ΔU = U(cycle 9) − U(cycle 0)  (mV)")
    a.set_title(f"{'bc'[j]}   {CHEM_LABEL[c]}: curve change by KP group")
    a.legend(fontsize=8.5)
fig.tight_layout(); save(fig, "kp_fig2_curves.png")

# ---------------------------------------------------------------- 1D CNN, repeated random 80/20 splits
splits = M.random_splits(n, REPEATS, seed=42)
oos = [[] for _ in range(n)]
cams = [[] for _ in range(n)]
sal = [[] for _ in range(n)]
split_rows = []
for r, (tr, te) in enumerate(splits):
    a, lo, hi = K.scale_pm1(X[tr]); b = K.scale_pm1(X[te], lo, hi)[0]
    P_te, P_tr, C_te, S_te = [], [], [], []
    for s in range(ENS):
        p = M.cnn_train(a, y[tr], seed=100 * r + s, **CFG)
        P_te.append(M.cnn_predict(p, b)); P_tr.append(M.cnn_predict(p, a))
        C_te.append(M.gradcam(p, b)); S_te.append(M.vanilla_saliency(p, b))
    yt, yr_ = np.mean(P_te, 0), np.mean(P_tr, 0)
    C_te, S_te = np.mean(C_te, 0), np.mean(S_te, 0)
    base = np.full(len(te), 10 / y[tr].mean())
    cbase = np.array([10 / y[tr][chem[tr] == c].mean() for c in chem[te]])
    split_rows.append({"train": M.mape(kp[tr], M.to_kp(yr_)), "test": M.mape(kp[te], M.to_kp(yt)),
                       "test_mae": M.mae(kp[te], M.to_kp(yt)), "mean_base": M.mape(kp[te], base),
                       "chem_base": M.mape(kp[te], cbase)})
    for k_, i in enumerate(te):
        oos[i].append(M.to_kp(yt[k_])); cams[i].append(C_te[k_]); sal[i].append(S_te[k_])
    print(r, {k: round(v, 1) for k, v in split_rows[-1].items()}, flush=True)

test = np.array([s["test"] for s in split_rows])
res["cnn"] = {k: {"mean": float(np.mean([s[k] for s in split_rows])), "sd": float(np.std([s[k] for s in split_rows])),
                  "min": float(np.min([s[k] for s in split_rows])), "max": float(np.max([s[k] for s in split_rows]))}
              for k in split_rows[0]}
res["cnn"]["splits_below_paper"] = int((test <= PAPER_TEST_MAPE).sum())
res["cnn"]["repeats"] = REPEATS
has = np.array([len(o) > 0 for o in oos])
kp_oos = np.array([np.mean(o) if o else np.nan for o in oos])
cam_cell = np.array([np.mean(c, 0) if c else np.full(201, np.nan) for c in cams])
sal_cell = np.array([np.mean(c, 0) if c else np.full(201, np.nan) for c in sal])
np.savez(OUT / "cnn_oos.npz", kp_oos=kp_oos, cam=cam_cell, sal=sal_cell, has=has)

# ---------------------------------------------------------------- Fig 3: CNN accuracy
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
for c in ["NMC", "S"]:
    mm = (chem == c) & has
    ax[0].plot(kp[mm], kp_oos[mm], "o", ms=5, color=CHEM_COLOR[c], alpha=0.85, label=CHEM_LABEL[c])
ax[0].plot([0, 75], [0, 75], ":", color=GREY)
ax[0].set_xlim(0, 75); ax[0].set_ylim(0, 75)
ax[0].set_xlabel("Actual KP (cycle)"); ax[0].set_ylabel("Predicted KP (mean when held out)")
ax[0].set_title(f"a   1D CNN, out-of-sample ({REPEATS} random 80/20 splits)"); ax[0].legend(fontsize=9)
cols = ["train", "test", "chem_base", "mean_base"]
labs = ["CNN train", "CNN test", "chemistry-mean\nbaseline", "overall-mean\nbaseline"]
vals = [[s[k] for s in split_rows] for k in cols]
bp = ax[1].boxplot(vals, tick_labels=labs, widths=0.5, patch_artist=True)
for patch, col in zip(bp["boxes"], [GREEN, BLUE, GREY, GREY]):
    patch.set_facecolor(col); patch.set_alpha(0.35)
for k_, v in enumerate(vals):
    ax[1].plot(np.full(len(v), k_ + 1) + np.random.default_rng(k_).uniform(-0.12, 0.12, len(v)), v, ".", color=NAVY, ms=4)
ax[1].axhline(PAPER_TEST_MAPE, color=RED, ls="--", lw=1.2); ax[1].text(4.45, PAPER_TEST_MAPE + 1, "paper test 15.2 %", color=RED, ha="right", fontsize=9)
ax[1].axhline(PAPER_TRAIN_MAPE, color=GREEN, ls=":", lw=1.2); ax[1].text(4.45, PAPER_TRAIN_MAPE - 3.5, "paper train 11.0 %", color=GREEN, ha="right", fontsize=9)
ax[1].set_ylabel("KP MAPE (%) per split"); ax[1].set_title("b   Error over repeated splits")
fig.tight_layout(); save(fig, "kp_fig3_cnn.png")

# ---------------------------------------------------------------- Fig 4: Grad-CAM by KP bin (paper Fig. 3E,F)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
res["gradcam_last10_share"] = {}
for j, c in enumerate(["NMC", "S"]):
    mm = (chem == c) & has
    edges = np.arange(10, 80, 10)
    rows, labs = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        g = mm & (kp >= lo_) & (kp < hi_)
        if g.sum() >= 2:
            rows.append(cam_cell[g].mean(0)); labs.append(f"KP {lo_}–{hi_ - 1} (n={g.sum()})")
    im = ax[j].imshow(np.array(rows), aspect="auto", cmap="magma", vmin=0, vmax=1, extent=[0, 1, len(rows) - 0.5, -0.5])
    ax[j].set_yticks(range(len(rows))); ax[j].set_yticklabels(labs, fontsize=9)
    dod_axis(ax[j]); ax[j].set_title(f"{'ab'[j]}   {CHEM_LABEL[c]}: mean Grad-CAM by KP bin")
    plt.colorbar(im, ax=ax[j], fraction=0.035, label="relevance (0–1)")
    res["gradcam_last10_share"][c] = float(cam_cell[mm][:, 181:].sum(1).mean() / cam_cell[mm].sum(1).mean())
fig.tight_layout(); save(fig, "kp_fig4_gradcam.png")

# ---------------------------------------------------------------- Fig 5: AV descriptor family
av = M.area_variation(X)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
res["av100"] = {}
for c in ["NMC", "S"]:
    mm = chem == c
    rho = spearmanr(av[mm, -1], kp[mm])[0]
    res["av100"][c] = float(rho)
    ax[0].loglog(av[mm, -1], kp[mm], "o", ms=5, color=CHEM_COLOR[c], alpha=0.85, label=f"{CHEM_LABEL[c]}   ρ = {rho:.2f}")
rho_all = spearmanr(av[:, -1], kp)[0]; res["av100"]["pooled"] = float(rho_all)
ax[0].set_xlabel("AV100 = mean area variation in the last 10 % DoD (V·point)"); ax[0].set_ylabel("KP (cycle)")
ax[0].set_title(f"a   KP vs AV100 (pooled ρ = {rho_all:.2f})"); ax[0].legend(fontsize=9)
x = np.arange(10)
res["rho_by_segment"] = {}
for k_, c in enumerate(["NMC", "S"]):
    mm = chem == c
    r_ = [spearmanr(av[mm, s], kp[mm])[0] for s in range(10)]
    res["rho_by_segment"][c] = [float(v) for v in r_]
    ax[1].bar(x + (k_ - 0.5) * 0.38, np.abs(r_), 0.38, color=CHEM_COLOR[c], label=CHEM_LABEL[c])
ax[1].set_xticks(x); ax[1].set_xticklabels([f"AV{10 * (s + 1)}" for s in range(10)], fontsize=9)
ax[1].set_ylabel("|Spearman ρ| with KP"); ax[1].set_title("b   Which DoD segment's change tracks KP?")
ax[1].legend(fontsize=9)
fig.tight_layout(); save(fig, "kp_fig5_av.png")

(OUT / "kp_repro.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
