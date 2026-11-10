"""Extension 3 — ideas from Attia et al., Nature 578, 397 (2020) (closed-loop optimisation).

(a) Uncertainty-guided labelling. Attia et al. let a Bayesian optimiser pick the next experiment by
    balancing exploration (high uncertainty) and exploitation. Here the "experiment" is a full
    charge-discharge to measure capacity for a spectrum we already have. The GPR (paper model)
    picks the spectrum it is least sure about; we compare with picking at random.
(b) Early prediction. Attia et al. replace a months-long test by predicting cycle life from the
    first cycles (elastic net on early-cycle features, Severson et al. 2019). Here: log(EoL cycle)
    from the change of the EIS spectrum over the first cycles, leave-one-cell-out over 11 cells.

Outputs fig/ext3a_active_labelling.png, fig/ext3b_early_prediction.png, fig/ext3_metrics.json
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import LeaveOneOut
import eis_data as D
import models as M
from common import stack, save, FIG, BLUE, RED, ORANGE, NAVY, GREY, GREEN, TRAIN_MT, TEST_25, TEST_MT

# ------------------------------------------------------------------ (a) active labelling
X, y, g, _ = stack(TRAIN_MT)
mu, sd = D.zscore_fit(X)
Z = (X - mu) / sd
Xt, yt, _, _ = stack(TEST_25 + TEST_MT)
Zt = (Xt - mu) / sd

BUDGET = list(range(10, 151, 10))
N_RUNS = 5


def run(strategy, seed):
    rng = np.random.default_rng(seed)
    lab = list(rng.choice(len(y), 10, replace=False))
    curve = []
    while True:
        m = M.gpr_se_iso().fit(Z[lab], y[lab])
        if len(lab) in BUDGET:
            curve.append(float(np.sqrt(np.mean((m.predict(Zt) - yt) ** 2))))
        if len(lab) >= BUDGET[-1]:
            return curve, lab
        pool = np.setdiff1d(np.arange(len(y)), lab)
        if strategy == "uncertainty":
            _, s = m.predict(Z[pool], return_std=True)
            new = pool[np.argsort(-s)[:5]]          # batch of 5, like a batch of parallel tests
        else:
            new = rng.choice(pool, 5, replace=False)
        lab += list(new)


curves, chosen = {}, {}
for strat in ["random", "uncertainty"]:
    out = [run(strat, s) for s in range(N_RUNS)]
    curves[strat] = np.array([o[0] for o in out])
    chosen[strat] = np.concatenate([o[1] for o in out])
full = M.gpr_se_iso().fit(Z, y)
rmse_full = float(np.sqrt(np.mean((full.predict(Zt) - yt) ** 2)))

soh_all = np.concatenate([D.paired(c)["cap_norm"] for c in TRAIN_MT])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for strat, col in [("random", GREY), ("uncertainty", BLUE)]:
    m_, s_ = curves[strat].mean(0), curves[strat].std(0)
    ax[0].fill_between(BUDGET, m_ - s_, m_ + s_, color=col, alpha=0.2, lw=0)
    ax[0].plot(BUDGET, m_, "-o", ms=4, color=col, label={"random": "Random spectra", "uncertainty": "Highest GPR uncertainty first"}[strat])
ax[0].axhline(rmse_full, color=NAVY, ls="--", lw=1, label=f"All {len(y)} labelled spectra")
ax[0].set_xlabel("Number of spectra with a measured capacity"); ax[0].set_ylabel("Test RMSE (mAh), 6 test cells")
ax[0].set_title("a   How many capacity measurements are needed?"); ax[0].legend()
bins = np.linspace(0.55, 1.05, 21)
ax[1].hist(soh_all, bins, color=GREY, alpha=0.35, density=True, label="All training spectra")
ax[1].hist(soh_all[chosen["uncertainty"]], bins, histtype="step", lw=2, color=BLUE, density=True, label="Picked by uncertainty")
ax[1].set_xlabel("SoH of the picked spectrum"); ax[1].set_ylabel("Density"); ax[1].legend()
ax[1].set_title("b   Where the uncertainty-guided picks fall")
fig.tight_layout(); save(fig, "ext3a_active_labelling.png")

# ------------------------------------------------------------------ (b) early prediction of EoL
EARLY = (2, 10)          # EIS cycles k=2 and k=10  (battery cycles 4 and 20)


def early_features(cell):
    d = D.paired(cell, truncate=False)
    a, b = EARLY
    Xa, Xb = d["X"][d["k"] == a][0], d["X"][d["k"] == b][0]
    dlt = Xb - Xa
    re_d, im_d = dlt[:60], dlt[60:]
    # Severson-style summaries of the change curve, plus the initial spectrum's key points
    return np.array([np.log10(np.var(re_d) + 1e-12), np.log10(np.var(im_d) + 1e-12),
                     np.mean(re_d), np.mean(im_d), np.min(im_d), np.max(im_d),
                     Xa[0], Xa[59], Xa[60 + 30], Xa[60 + 39]])


names = ["log var ΔRe", "log var ΔIm", "mean ΔRe", "mean ΔIm", "min ΔIm", "max ΔIm",
         "Re 20 kHz", "Re 0.02 Hz", "−Im 17.8 Hz", "−Im 2.16 Hz"]
cells, F, eol, temp = [], [], [], []
for c in D.CELLS_ALL:
    r = D.rul_labels(D.paired(c, truncate=False))
    if r is None:
        continue
    cells.append(c); F.append(early_features(c)); eol.append(r["eol_cycle"]); temp.append(int(c[:2]))
F, eol, temp = np.array(F), np.array(eol, float), np.array(temp)
Fz = (F - F.mean(0)) / F.std(0)
pred = np.zeros(len(eol))
for tr, te in LeaveOneOut().split(Fz):
    mu_f, sd_f = F[tr].mean(0), F[tr].std(0)
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=min(5, len(tr)), max_iter=20000)
    en.fit((F[tr] - mu_f) / sd_f, np.log10(eol[tr]))
    pred[te] = 10 ** en.predict((F[te] - mu_f) / sd_f)
mape = float(np.mean(np.abs(pred - eol) / eol) * 100)
base = np.array([10 ** np.mean(np.log10(np.delete(eol, i))) for i in range(len(eol))])
mape_base = float(np.mean(np.abs(base - eol) / eol) * 100)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
cmap = {25: BLUE, 35: ORANGE, 45: RED}
for c, t, e, p in zip(cells, temp, eol, pred):
    ax[0].plot(e, p, "o", ms=8, color=cmap[t]); ax[0].annotate(c, (e, p), textcoords="offset points", xytext=(5, 3), fontsize=8)
lim = [8, max(600, float(pred.max()) * 1.4)]; ax[0].plot(lim, lim, ":", color=GREY)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Actual EoL (cycles)"); ax[0].set_ylabel("Predicted EoL (cycles, leave-one-cell-out)")
ax[0].set_title(f"a   EoL from cycles 4 → 20 EIS change   MAPE {mape:.0f} %  (mean-only baseline {mape_base:.0f} %)", fontsize=10)
en_all = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=20000).fit(Fz, np.log10(eol))
ax[1].barh(names, en_all.coef_, color=[BLUE if v < 0 else RED for v in en_all.coef_])
ax[1].axvline(0, color=GREY, lw=0.8); ax[1].invert_yaxis()
ax[1].set_xlabel("Elastic-net coefficient on log10 EoL (all 11 cells, standardised features)")
ax[1].set_title("b   Which early features carry the signal")
fig.tight_layout(); save(fig, "ext3b_early_prediction.png")

out = {"active": {"budget": BUDGET, "random_mean": curves["random"].mean(0).round(3).tolist(),
                  "uncertainty_mean": curves["uncertainty"].mean(0).round(3).tolist(), "all_labels_rmse": round(rmse_full, 3),
                  "n_pool": int(len(y))},
       "early": {"cells": cells, "eol": eol.tolist(), "pred": pred.round(1).tolist(), "mape_%": round(mape, 1),
                 "baseline_mape_%": round(mape_base, 1), "coef": dict(zip(names, en_all.coef_.round(4).tolist()))}}
(FIG / "ext3_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
