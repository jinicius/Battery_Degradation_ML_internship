"""Extension 2 — inter-cell learning (BatLiNet, Zhang et al., Nat. Mach. Intell. 7, 270 (2025)).

Idea: besides learning y = f(x) for each cell ("intra-cell"), learn how the DIFFERENCE between two
spectra maps to the DIFFERENCE between their targets, Δy = g(x_i − x_j). At test time a reference
spectrum x' with known y' gives y ≈ g(x − x') + y'; averaging over many references and blending with
the intra-cell prediction stabilises the estimate when ageing conditions (here: temperature) differ.

Task: RUL from one state-V EIS spectrum, the multi-temperature setting of the paper's Fig. 4.
Train cells: 25C01–03, 35C01, 45C01 (rows up to EoL).  Test: 25C05–08, 35C02, 45C02.

Outputs fig/ext2a_intercell_rul.png, fig/ext2b_intercell_scatter.png, fig/ext2_metrics.json
"""
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import eis_data as D
import models as M
import cnn_jax as C
from common import rul_stack, stack, save, FIG, BLUE, RED, GREEN, ORANGE, NAVY, GREY, TRAIN_MT, TEST_25, TEST_MT

TEST = TEST_25 + TEST_MT
Xc, _, _, _ = stack(TRAIN_MT)
mu, sd = D.zscore_fit(Xc)                     # same normalisation as the paper's Fig. 4 model
Xr, yr, gr = rul_stack(TRAIN_MT)
Zr = (Xr - mu) / sd
yscale = yr.std()


def test_set(cell):
    d = D.paired(cell, truncate=False)
    r = D.rul_labels(d)
    return (d["X"][r["idx"]] - mu) / sd, r["rul"].astype(float)


# ---- baseline: paper's linear-kernel GPR
lin = M.gpr_lin().fit(Zr, yr)

# ---- intra-cell CNN ensemble
SEEDS = range(4)
intra = [C.train(C.to_seq(Zr), yr / yscale, seed=s, epochs=250)[0] for s in SEEDS]


# ---- inter-cell CNN: trained on random pairs (x_i - x_j -> y_i - y_j), pairs across different cells too
def train_inter(seed, epochs=250, pairs=2048, lr=2e-3, wd=1e-4):
    key = jax.random.PRNGKey(100 + seed)
    p = C.init_params(key)
    opt = optax.adamw(lr, weight_decay=wd); st = opt.init(p)
    S = jnp.asarray(C.to_seq(Zr)); Y = jnp.asarray(yr / yscale)

    @jax.jit
    def step(p, st, i, j):
        def loss(p):
            return jnp.mean((C.forward(p, S[i] - S[j]) - (Y[i] - Y[j])) ** 2)
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    n = len(yr)
    for ep in range(epochs):
        for _ in range(4):
            i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
            p, st, l = step(p, st, jnp.asarray(i), jnp.asarray(j))
    return p


inter = [train_inter(s) for s in SEEDS]
REF = np.random.default_rng(7).choice(len(yr), 256, replace=False)      # reference spectra (training rows)


def pred_intra(Zt):
    return np.mean([C.predict(p, C.to_seq(Zt)) for p in intra], 0) * yscale


def pred_inter(Zt):
    S = C.to_seq(Zt); R = C.to_seq(Zr[REF])
    out = np.zeros(len(Zt))
    for p in inter:
        diff = (S[:, None] - R[None]).reshape(-1, 60, 2)
        dy = C.predict(p, diff).reshape(len(Zt), len(REF)) * yscale
        out += (dy + yr[REF][None]).mean(1)
    return out / len(inter)


ALPHA = 0.5                                   # BatLiNet blend: y = a*y_intra + (1-a)*y_inter
rows, preds = {}, {}
for c in TEST:
    Zt, rt = test_set(c)
    pl = lin.predict(Zt); pi = pred_intra(Zt); pc = pred_inter(Zt); pb = ALPHA * pi + (1 - ALPHA) * pc
    preds[c] = (rt, pl, pi, pc, pb)
    rmse = lambda p: float(np.sqrt(np.mean((rt - p) ** 2)))
    rows[c] = {"GPR-LIN (paper)": rmse(pl), "CNN intra": rmse(pi), "CNN inter": rmse(pc), "Blend α=0.5": rmse(pb),
               "R2": {"GPR-LIN (paper)": round(M.r2(rt, pl), 3), "CNN intra": round(M.r2(rt, pi), 3),
                      "CNN inter": round(M.r2(rt, pc), 3), "Blend α=0.5": round(M.r2(rt, pb), 3)}}

names = ["GPR-LIN (paper)", "CNN intra", "CNN inter", "Blend α=0.5"]
cols = [GREY, ORANGE, BLUE, NAVY]
fig, ax = plt.subplots(figsize=(12, 4.6))
x = np.arange(len(TEST)); w = 0.2
for k, (n_, col) in enumerate(zip(names, cols)):
    ax.bar(x + (k - 1.5) * w, [rows[c][n_] for c in TEST], w, color=col, label=n_)
ax.set_xticks(x); ax.set_xticklabels(TEST); ax.set_ylabel("RUL RMSE (cycles)  — lower is better")
mean_r = {n_: float(np.mean([rows[c][n_] for c in TEST])) for n_ in names}
ax.set_title("RUL from one EIS spectrum, train 25/35/45 °C — mean RMSE: " +
             ",  ".join(f"{n_} {v:.0f}" for n_, v in mean_r.items()), fontsize=10)
ax.legend(ncol=4, loc="upper left")
save(fig, "ext2a_intercell_rul.png")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, c in zip(ax, ["25C05", "35C02", "45C02"]):
    rt, pl, pi, pc, pb = preds[c]
    a.plot(rt, pl, ".", ms=4, color=GREY, label=f"GPR-LIN  R²={rows[c]['R2']['GPR-LIN (paper)']:.2f}")
    a.plot(rt, pi, ".", ms=4, color=ORANGE, label=f"CNN intra  R²={rows[c]['R2']['CNN intra']:.2f}")
    a.plot(rt, pb, ".", ms=4, color=NAVY, label=f"Blend  R²={rows[c]['R2']['Blend α=0.5']:.2f}")
    lim = [0, rt.max() * 1.1]; a.plot(lim, lim, ":", color=GREY, lw=0.8)
    a.set_xlabel("Actual RUL"); a.set_ylabel("Predicted RUL"); a.set_title(c); a.legend(fontsize=8.5)
fig.tight_layout(); save(fig, "ext2b_intercell_scatter.png")

out = {"rmse_cycles": {c: {k: round(v, 1) for k, v in r.items() if k != "R2"} for c, r in rows.items()},
       "r2": {c: r["R2"] for c, r in rows.items()}, "mean_rmse": {k: round(v, 1) for k, v in mean_r.items()},
       "n_references": len(REF), "alpha": ALPHA, "train_rows": int(len(yr))}
(FIG / "ext2_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
