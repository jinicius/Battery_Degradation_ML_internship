"""Shared plotting style and dataset builders."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eis_data as D

FIG = Path(__file__).resolve().parents[1] / "fig"
FIG.mkdir(exist_ok=True)

BLUE, RED, GREEN, PINK, LGREEN = "#0082D8", "#CD2746", "#77AC2D", "#FFBFC8", "#C1DDC6"
NAVY, ORANGE, GREY = "#1E2A47", "#E08A1E", "#8A8F99"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})

TRAIN_25 = ["25C01", "25C02", "25C03", "25C04"]
TEST_25 = ["25C05", "25C06", "25C07", "25C08"]
TRAIN_MT = TRAIN_25 + ["35C01", "45C01"]
TEST_MT = ["35C02", "45C02"]


def stack(cells, key="cap"):
    ds = [D.paired(c) for c in cells]
    X = np.vstack([d["X"] for d in ds])
    y = np.concatenate([d[key] for d in ds])
    g = np.concatenate([[d["cell"]] * len(d["k"]) for d in ds])
    return X, y, g, ds


def rul_stack(cells):
    Xs, ys, gs = [], [], []
    for c in cells:
        d = D.paired(c, truncate=False)
        r = D.rul_labels(d)
        if r is None:
            continue
        Xs.append(d["X"][r["idx"]]); ys.append(r["rul"]); gs += [c] * len(r["idx"])
    return np.vstack(Xs), np.concatenate(ys), np.array(gs)


def feature_labels(freq):
    return [f"Re {f:.3g} Hz" for f in freq] + [f"Im {f:.3g} Hz" for f in freq]


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    return p
