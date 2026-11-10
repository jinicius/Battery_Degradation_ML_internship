"""
Data loader for Liu et al., JACS 146, 33012 (2024) — knee-point (KP) prediction for Li metal batteries.
Raw data: github.com/xinyanl/SELF  (NCM_1/2.zip: 33 Li|NMC cells, S_1..4.zip: 46 Li|S cells)

CSV columns: cycle_number (0-based), discharge_cycle (= cycle + normalised discharge capacity, i.e. DoD
fraction in [0,1] inside each cycle), U (V), tot_capacity (mAh, cumulative within the discharge).

Outputs (cached in data/kp_dataset.npz):
  X      (n_cell, 201, 10)  discharge voltage of cycles 0..9 resampled on a DoD grid of 201 points
                            -> the paper's input matrix [V_{i,j}], i = time step (DoD), j = cycle
  Q      list of per-cycle discharge capacity trajectories
  KP     knee point (cycle index) from LOESS smoothing + Kneedle (max curvature), as in the paper's code
  chem   'NMC' or 'S';  batch = folder + date token (cells assembled in the same batch)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
CACHE = ROOT / "data" / "kp_dataset.npz"
N_PT, N_CYC = 201, 10
DOD = np.linspace(0.0, 1.0, N_PT)

# Table S1 of the SI (knee points reported by the authors), used only to check our KP extraction
TABLE_S1_KP = {
    "NMC": [45, 30, 26, 57, 57, 47, 44, 43, 42, 36, 59, 44, 59, 56, 54, 54, 53, 52, 52, 46, 40, 39, 37,
            33, 31, 31, 26, 25, 22, 18, 15, 44, 29],
    "S": [18, 15, 14, 35, 29, 17, 12, 26, 24, 24, 40, 31, 55, 21, 12, 32, 20, 27, 16, 25, 22, 14, 16,
          25, 22, 45, 38, 37, 53, 46, 42, 42, 27, 35, 25, 23, 56, 30, 22, 21, 56, 34, 11, 29, 14, 11],
}


def cell_files():
    fs = sorted(RAW.glob("*/*.csv"))
    out = []
    for f in fs:
        folder = f.parent.name                     # NCM_1, NCM_2, S_1 ...
        stem = f.stem
        chem = "NMC" if folder.startswith("NCM") else "S"
        parts = [p for p in stem.split("_") if p]
        date = parts[1] if len(parts) >= 3 else "nodate"
        out.append(dict(path=f, name=f"{folder}/{stem}", chem=chem, batch=f"{folder}:{date}"))
    return out


def read_cell(path):
    d = pd.read_csv(path, usecols=["cycle_number", "discharge_cycle", "U", "tot_capacity"])
    return d


def capacity_trajectory(d):
    return d.groupby("cycle_number").tot_capacity.max().sort_index().values


def resample_cycles(d, n_cycle=N_CYC):
    curves = []
    for c in range(n_cycle):
        g = d[d.cycle_number == c]
        frac = g.discharge_cycle.values - np.floor(g.discharge_cycle.values.min())
        o = np.argsort(frac, kind="stable")
        frac, u = frac[o], g.U.values[o]
        frac, idx = np.unique(frac, return_index=True)
        curves.append(np.interp(DOD, frac, u[idx]))
    return np.stack(curves, axis=1)               # (201, n_cycle)


def knee_point(q, ref="first", frac=0.25):
    """The paper's get_knee_point(): retention > 0.75 -> LOESS(frac=0.25) -> Kneedle(concave, decreasing)."""
    import statsmodels.api as sm
    from kneed import KneeLocator
    r = q / (q[0] if ref == "first" else np.max(q[:5]))
    cyc = np.arange(len(q))
    keep = r > 0.75
    rr, cc = r[keep], cyc[keep]
    sm_ = sm.nonparametric.lowess(rr, cc, frac=frac)[:, 1]
    idx = np.arange(len(sm_))                     # the authors use the positional index of the kept rows
    k = KneeLocator(idx, sm_, S=1, curve="concave", direction="decreasing", interp_method="interp1d").elbow
    return (int(k) if k is not None else None), sm_, cc


def build(ref="first", force=False):
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    rows = cell_files()
    X, KP, Q, names, chem, batch = [], [], [], [], [], []
    for r in rows:
        d = read_cell(r["path"])
        q = capacity_trajectory(d)
        k, _, _ = knee_point(q, ref=ref)
        X.append(resample_cycles(d)); KP.append(k); Q.append(q)
        names.append(r["name"]); chem.append(r["chem"]); batch.append(r["batch"])
    out = dict(X=np.array(X), KP=np.array(KP, dtype=float), Q=np.array(Q, dtype=object),
               names=np.array(names), chem=np.array(chem), batch=np.array(batch), dod=DOD)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    return out


def scale_pm1(X, lo=None, hi=None):
    """Rescale voltages to [-1, 1] (SI 1.4). One global min/max taken from the training cells."""
    lo = X.min() if lo is None else lo
    hi = X.max() if hi is None else hi
    return 2 * (X - lo) / (hi - lo) - 1, lo, hi
