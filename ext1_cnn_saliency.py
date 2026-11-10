"""Extension 1 — 1D CNN + saliency map (Liu et al., JACS 2024) applied to EIS spectra.

Outputs
  fig/ext1a_cnn_vs_gpr.png        capacity estimation on test cells, CNN ensemble vs GPR (paper model)
  fig/ext1b_saliency_spectrum.png saliency over frequency (Re, -Im) with ARD weights for comparison
  fig/ext1c_saliency_by_stage.png saliency averaged by state of health (early / mid / late life)
  fig/ext1d_nyquist_saliency.png  Nyquist plots coloured by saliency
  fig/ext1e_descriptor.png        a single EIS descriptor suggested by the saliency map vs SoH
  fig/ext1_metrics.json
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import eis_data as D
import models as M
import cnn_jax as C
from common import stack, save, FIG, BLUE, RED, GREEN, NAVY, ORANGE, GREY, TRAIN_MT, TEST_25, TEST_MT

SEEDS = range(5)
TEST = TEST_25 + TEST_MT

X, y, g, _ = stack(TRAIN_MT)
mu, sd = D.zscore_fit(X)
ym, ys = y.mean(), y.std()
Z = C.to_seq((X - mu) / sd)
rng = np.random.default_rng(0)
val = rng.random(len(y)) < 0.15                      # random 15 % of training rows for early stopping

models = []
for s in SEEDS:
    p, _ = C.train(Z[~val], (y[~val] - ym) / ys, Z[val], (y[val] - ym) / ys, seed=s, epochs=300)
    models.append(p)

iso = M.gpr_se_iso().fit((X - mu) / sd, y)


def cnn_pred(Xr):
    Zt = C.to_seq((Xr - mu) / sd)
    P = np.array([C.predict(p, Zt) for p in models]) * ys + ym
    return P.mean(0), P.std(0)


def cnn_sal(Xr):
    Zt = C.to_seq((Xr - mu) / sd)
    return np.mean([C.saliency(p, Zt) for p in models], axis=0)   # (n, 60, 2), d(std. cap)/d(std. input)


FREQ = D.FREQ
res = {}
for c in TEST:
    d = D.paired(c)
    m, s = cnn_pred(d["X"])
    gm = iso.predict((d["X"] - mu) / sd)
    res[c] = dict(d=d, cnn=m, cnn_sd=s, gpr=gm, r2_cnn=M.r2(d["cap"], m), r2_gpr=M.r2(d["cap"], gm),
                  rmse_cnn=float(np.sqrt(np.mean((d["cap"] - m) ** 2))), rmse_gpr=float(np.sqrt(np.mean((d["cap"] - gm) ** 2))),
                  sal=cnn_sal(d["X"]))

# ---- a. predictions
fig, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
for a, c in zip(ax, ["25C05", "25C06", "25C07", "35C02", "45C02", "25C08"]):
    r = res[c]; x = r["d"]["battery_cycle"]
    a.fill_between(x, r["cnn"] - r["cnn_sd"], r["cnn"] + r["cnn_sd"], color="#F6D6A8", lw=0)
    a.plot(x, r["d"]["cap"], "x", ms=3, color=BLUE, label="Measured")
    a.plot(x, r["gpr"], "-", color=RED, lw=1.2, label=f"GPR (paper)  R²={r['r2_gpr']:.2f}")
    a.plot(x, r["cnn"], "-", color=ORANGE, lw=1.6, label=f"1D CNN ×5  R²={r['r2_cnn']:.2f}")
    a.set_title(c); a.set_xlabel("Cycle number"); a.set_ylabel("Capacity (mAh)"); a.legend(fontsize=8.5, loc="upper right")
fig.suptitle("Capacity from one EIS spectrum — trained on 25C01–04, 35C01, 45C01", y=1.01)
fig.tight_layout(); save(fig, "ext1a_cnn_vs_gpr.png")

# ---- b. saliency spectrum vs ARD
S_all = np.concatenate([np.abs(res[c]["sal"]) for c in TEST])          # (N, 60, 2)
s_mean = S_all.mean(0)
s_norm = s_mean / s_mean.sum()
try:
    from gp_jax import fit_ard_restarts
    best, _ = fit_ard_restarts((X - mu) / sd, y, n_restarts=4, seed=3)
    w = best.ard_weights()
except Exception:
    w = None
fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
for j, (name, col) in enumerate([("Re(Z)", BLUE), ("−Im(Z)", RED)]):
    ax[j].semilogx(FREQ, s_norm[:, j], "-o", ms=3.5, color=col, label="CNN saliency |∂ŷ/∂x| (mean over test spectra)")
    if w is not None:
        ax[j].semilogx(FREQ, w[j * 60:(j + 1) * 60] * s_norm.max() / max(w.max(), 1e-12), "s", ms=4, mfc="none",
                       color=GREY, label="ARD weight (same training set, rescaled)")
    for f0, ls in [(17.8, "--"), (2.16, ":")]:
        ax[j].axvline(f0, color=NAVY, lw=0.8, ls=ls)
    ax[j].set_ylabel(f"{name} relevance"); ax[j].set_title(f"{name} channel"); ax[j].legend(fontsize=9)
ax[1].set_xlabel("Frequency (Hz)"); ax[1].invert_xaxis()
fig.suptitle("Which frequencies does the model use?  dashed 17.8 Hz, dotted 2.16 Hz (paper's ARD picks)", y=1.0)
fig.tight_layout(); save(fig, "ext1b_saliency_spectrum.png")

# ---- c. saliency by life stage (signed), all test cells pooled
soh = np.concatenate([res[c]["d"]["cap_norm"] for c in TEST])
S_signed = np.concatenate([res[c]["sal"] for c in TEST])
bins = [(0.9, 1.01, "SoH > 90 %"), (0.8, 0.9, "80–90 %"), (0.0, 0.8, "< 80 % (past EoL)")]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for j, name in enumerate(["Re(Z)", "−Im(Z)"]):
    M_ = np.array([S_signed[(soh >= lo) & (soh < hi), :, j].mean(0) for lo, hi, _ in bins])
    v = np.abs(M_).max()
    im = ax[j].imshow(M_, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                      extent=[np.log10(FREQ[0]), np.log10(FREQ[-1]), len(bins) - 0.5, -0.5])
    ax[j].set_yticks(range(len(bins))); ax[j].set_yticklabels([b[2] for b in bins])
    ax[j].set_xlabel("log10 frequency (Hz)"); ax[j].set_title(f"{name}: signed saliency ∂ŷ/∂x")
    plt.colorbar(im, ax=ax[j], fraction=0.04)
fig.suptitle("Saliency averaged by life stage (red = larger value here raises the predicted capacity)", y=1.02)
fig.tight_layout(); save(fig, "ext1c_saliency_by_stage.png")

# ---- d. Nyquist coloured by saliency
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, c in zip(ax, ["25C05", "35C02", "45C02"]):
    r = res[c]; d = r["d"]; n = len(d["k"])
    sal = np.abs(r["sal"]).sum(2)                    # (n, 60)
    vmax = np.percentile(sal, 99)
    picks = np.linspace(0, n - 1, 5).astype(int)
    for i in picks:
        re, im = d["X"][i, :60], d["X"][i, 60:]
        pts = np.c_[re, im][:, None, :]
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, vmax), lw=3)
        lc.set_array(sal[i, :-1]); a.add_collection(lc)
        if i in (picks[0], picks[-1]):
            a.annotate(f"cycle {d['battery_cycle'][i]}", (re[-1], im[-1]), textcoords="offset points",
                       xytext=(4, 6 if i == picks[-1] else -12), fontsize=8.5, color=GREY)
    a.autoscale()
    a.set_xlabel("Re(Z) (Ω)"); a.set_ylabel("−Im(Z) (Ω)"); a.set_title(f"{c}: 5 spectra across life")
fig.colorbar(lc, ax=ax, fraction=0.02, label="|saliency| (Re + Im)")
save(fig, "ext1d_nyquist_saliency.png")

# ---- e. single descriptor from the saliency map (cf. AV100 descriptor in the LMB paper)
top = np.argsort(-s_norm.sum(1))[:3]                 # three most salient frequencies (both channels)
band = sorted(top)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
desc = {}
for c in D.CELLS_ALL:
    d = D.paired(c)
    v_sal = d["X"][:, 60:][:, band].mean(1)           # mean −Im(Z) over the salient frequencies
    v_ohm = d["X"][:, 0]                              # Re(Z) at 20 kHz ≈ ohmic resistance
    desc[c] = (v_sal, v_ohm, d["cap_norm"])
cols = {"25": BLUE, "35": ORANGE, "45": RED}
for c, (vs, vo, soh_c) in desc.items():
    ax[0].plot(vs, soh_c, ".", ms=3, color=cols[c[:2]], alpha=0.7)
    ax[1].plot(vo, soh_c, ".", ms=3, color=cols[c[:2]], alpha=0.7)
allv = np.concatenate([v[0] for v in desc.values()]); allo = np.concatenate([v[1] for v in desc.values()])
alls = np.concatenate([v[2] for v in desc.values()])
rho_s = float(np.corrcoef(allv, alls)[0, 1]); rho_o = float(np.corrcoef(allo, alls)[0, 1])
fl = ", ".join(f"{FREQ[i]:.3g}" for i in band)
ax[0].set_xlabel(f"mean −Im(Z) at {fl} Hz (Ω)"); ax[0].set_title(f"Salient-band descriptor   r = {rho_s:.2f}")
ax[1].set_xlabel("Re(Z) at 20 kHz (Ω)  ≈ ohmic"); ax[1].set_title(f"High-frequency resistance   r = {rho_o:.2f}")
for a in ax:
    a.set_ylabel("SoH (capacity / initial)")
for t, col in cols.items():
    ax[0].plot([], [], "o", color=col, label=f"{t} °C")
ax[0].legend()
fig.suptitle("All 12 cells — one number from the EIS spectrum vs state of health", y=1.02)
fig.tight_layout(); save(fig, "ext1e_descriptor.png")

out = {c: {k: round(float(v), 4) for k, v in r.items() if k.startswith(("r2", "rmse"))} for c, r in res.items()}
out["salient_freqs_Hz"] = [round(float(FREQ[i]), 3) for i in band]
out["top10_saliency"] = [{"part": "Re" if j == 0 else "Im", "freq_Hz": round(float(FREQ[i]), 3), "share": round(float(s_norm[i, j]), 4)}
                         for i, j in zip(*np.unravel_index(np.argsort(-s_norm, axis=None)[:10], s_norm.shape))]
out["corr_descriptor_vs_soh"] = {"salient_band_-Im": round(rho_s, 3), "Re_20kHz": round(rho_o, 3)}
np.savez(FIG / "ext1_arrays.npz", saliency=s_norm.T.ravel(), ard=(w if w is not None else np.zeros(120)), freq=FREQ)
(FIG / "ext1_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
