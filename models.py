"""GPR models of Zhang et al. (2020) in scikit-learn.

MATLAB/GPML original -> scikit-learn here
  covSEard  + likGauss  -> ConstantKernel * RBF(length_scale=[120]) + WhiteKernel   (EIS-Capacity, ARD)
  covSEiso  + likGauss  -> ConstantKernel * RBF(length_scale=scalar) + WhiteKernel  (multi-T EIS-Capacity)
  covLINiso + likGauss  -> ConstantKernel * DotProduct(sigma_0=0) + WhiteKernel     (EIS-RUL)
Zero mean function in every case (normalize_y=False). Hyper-parameters by maximising
the log marginal likelihood, as in the paper.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct


def gpr_se_ard(n_features, restarts=0, seed=0):
    k = (C(1.0, (1e-3, 1e5)) * RBF(np.ones(n_features), (1e-2, 1e5))
         + WhiteKernel(0.01, (1e-6, 1e1)))
    return GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=restarts, random_state=seed)


def gpr_se_iso(restarts=0, seed=0):
    k = C(1.0, (1e-3, 1e5)) * RBF(1.0, (1e-2, 1e5)) + WhiteKernel(0.01, (1e-6, 1e1))
    return GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=restarts, random_state=seed)


def gpr_lin(restarts=0, seed=0):
    k = C(1.0, (1e-4, 1e6)) * DotProduct(sigma_0=0.0, sigma_0_bounds="fixed") + WhiteKernel(1.0, (1e-6, 1e4))
    return GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=restarts, random_state=seed)


def ard_weights(gpr):
    """Predictor weight w_m = exp(-l_m), normalised to sum 1 (paper Methods; authors' ARD_GPR.m)."""
    ls = np.asarray(gpr.kernel_.k1.k2.length_scale)
    w = np.exp(-ls)
    return w / w.sum(), ls


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
