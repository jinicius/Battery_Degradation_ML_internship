"""
EIS / Capacity raw-data loader for Zhang et al., Nat. Commun. 11, 1706 (2020).

Raw files (Zenodo 10.5281/zenodo.3633835):
  EIS_state_V_<cell>.txt   : one row per frequency point, 60 points per EIS cycle
                             columns = time, cycle number, freq, Re(Z), -Im(Z), |Z|, phase
  Data_Capacity_<cell>.txt : time, cycle number, ox/red (1 = charge, 0 = discharge), Capacity/mAh

Pairing rule (checked against the authors' preprocessed MATLAB matrices):
  EIS cycle k (1-based)  <->  discharge capacity of capacity-cycle k-1
  feature vector x_k = [Re(Z) at 60 freqs, -Im(Z) at 60 freqs]  (120 features)
A spectrum with a missing frequency point (59 points) is interpolated onto the
60-point grid in log10(f).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EIS_DIR = ROOT / "data" / "eis"
CAP_DIR = ROOT / "data" / "cap"

CELLS_25 = [f"25C0{i}" for i in range(1, 9)]
CELLS_ALL = CELLS_25 + ["35C01", "35C02", "45C01", "45C02"]

# The authors' MATLAB matrices pair EIS 25C02 with capacity file 25C03 and vice versa
# (checked row by row against Capacity_data.txt / RUL.txt). We follow that pairing.
CAP_FILE_FOR = {"25C02": "25C03", "25C03": "25C02"}

# Truncation used in the authors' MATLAB matrices (EIS_data.txt has 1358 rows)
MAX_EIS_CYCLES = {"25C01": 200, "35C01": 299, "35C02": 299, "45C01": 299, "45C02": 299}


def _read_eis(cell, state="V"):
    return pd.read_csv(EIS_DIR / f"EIS_state_{state}_{cell}.txt", sep=r"\s+", skiprows=1,
                       header=None, names=["t", "cyc", "f", "re", "im", "absz", "ph"])


def frequency_grid(cell="25C01", state="V"):
    e = _read_eis(cell, state)
    g = e.groupby("cyc")
    first_full = next(c for c, d in g if len(d) == 60)
    return e[e.cyc == first_full].f.values  # 20 kHz -> 0.02 Hz, descending


FREQ = None


def load_eis(cell, state="V"):
    """Return (cycles[int], X[n, 120], freq[60])."""
    global FREQ
    if FREQ is None:
        FREQ = frequency_grid()
    e = _read_eis(cell, state)
    lf = np.log10(FREQ)[::-1]
    cyc, rows = [], []
    for c, g in e.groupby("cyc"):
        if len(g) == 60:
            re, im = g.re.values, g.im.values
        else:  # interpolate onto the common grid
            x = np.log10(g.f.values)[::-1]
            re = np.interp(lf, x, g.re.values[::-1])[::-1]
            im = np.interp(lf, x, g.im.values[::-1])[::-1]
        cyc.append(int(c))
        rows.append(np.r_[re, im])
    return np.array(cyc), np.array(rows), FREQ


def load_capacity(cell):
    """Discharge capacity (mAh) per capacity-cycle index 0..N-1."""
    path = CAP_DIR / f"Data_Capacity_{cell}.txt"
    # Column layout differs between cells: 4 columns (time, cycle, ox/red, capacity)
    # or 6 columns (time, cycle, ox/red, Ewe, I, capacity). Capacity is always last.
    with open(path, encoding="latin-1") as fh:
        fh.readline()
        ncol = len(fh.readline().split())
    c = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, usecols=[1, 2, ncol - 1])
    c.columns = ["cyc", "ox", "cap"]
    d = c[c.ox == 0].groupby("cyc").cap.max()
    return d.sort_index().values


def paired(cell, state="V", truncate=True):
    """EIS features and the matching discharge capacity for one cell.

    Returns dict with cycles (EIS index k), battery_cycle (=2k), X, cap, cap_norm.
    """
    k, X, f = load_eis(cell, state)
    cap = load_capacity(CAP_FILE_FOR.get(cell, cell))
    n = min(len(k), len(cap))  # some capacity files stop before the EIS files do
    if truncate and cell in MAX_EIS_CYCLES:
        n = min(n, MAX_EIS_CYCLES[cell])
    k, X, cap = k[:n], X[:n], cap[k[:n] - 1]
    return dict(cell=cell, k=k, battery_cycle=2 * k, X=X, cap=cap, cap_norm=cap / cap[0], freq=f)


def eol_index(cap, threshold=0.8):
    """First EIS index whose capacity falls below threshold x initial; None if never."""
    below = np.where(cap < threshold * cap[0])[0]
    return int(below[0]) if len(below) else None


def rul_labels(d, threshold=0.8):
    """RUL in battery cycles for rows up to and including EoL (RUL = 2*(k_eol - k))."""
    i = eol_index(d["cap"], threshold)
    if i is None:
        return None
    k = d["k"][: i + 1]
    return dict(idx=np.arange(i + 1), rul=2 * (k[i] - k), eol_cycle=int(2 * k[i]))


def zscore_fit(X):
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    return mu, sd
