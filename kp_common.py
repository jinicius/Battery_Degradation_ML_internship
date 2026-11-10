"""Plot style and shared helpers for the KP study (same palette as the EIS study)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parents[1] / "fig"
FIG.mkdir(exist_ok=True)
OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)

BLUE, RED, GREEN, NAVY, ORANGE, GREY, PURPLE = "#0082D8", "#CD2746", "#77AC2D", "#1E2A47", "#E08A1E", "#8A8F99", "#7B3FA0"
CHEM_COLOR = {"NMC": BLUE, "S": ORANGE}
CHEM_LABEL = {"NMC": "Li|NMC (carbonate)", "S": "Li|S (ether)"}
PAPER_TEST_MAPE = 15.2
PAPER_TRAIN_MAPE = 11.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    return p


def dod_axis(ax):
    ax.set_xlabel("Depth of discharge (normalised capacity)")
    ax.set_xlim(0, 1)
