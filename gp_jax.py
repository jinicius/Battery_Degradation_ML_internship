"""Exact GP regression with an ARD squared-exponential kernel, written in JAX.

Same model as GPML covSEard + likGauss + meanZero (Zhang et al. 2020, Methods eq. 4):
    k(x, x') = sf^2 * exp(-0.5 * sum_m (x_m - x'_m)^2 / l_m^2),   y = f(x) + N(0, sn^2)
Hyper-parameters (log l_1..l_d, log sf, log sn) maximise the log marginal likelihood
with L-BFGS-B. JAX gives the exact gradient, which is far faster than building the
n x n x d gradient tensor scikit-learn uses for 120 length scales.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def _k(Xa, Xb, logl, logsf):
    A, B = Xa / jnp.exp(logl), Xb / jnp.exp(logl)
    d2 = jnp.sum(A**2, 1)[:, None] + jnp.sum(B**2, 1)[None, :] - 2 * A @ B.T
    return jnp.exp(2 * logsf) * jnp.exp(-0.5 * jnp.maximum(d2, 0.0))


def _nlml(theta, X, y, d):
    logl, logsf, logsn = theta[:d], theta[d], theta[d + 1]
    n = X.shape[0]
    K = _k(X, X, logl, logsf) + (jnp.exp(2 * logsn) + 1e-8) * jnp.eye(n)
    L = jnp.linalg.cholesky(K)
    a = jax.scipy.linalg.cho_solve((L, True), y)
    return 0.5 * y @ a + jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * n * jnp.log(2 * jnp.pi)


class ARDGP:
    def __init__(self, init_l=1.0, init_sf=1.0, init_sn=0.1, maxiter=300):
        self.init = (init_l, init_sf, init_sn)
        self.maxiter = maxiter

    def fit(self, X, y):
        X, y = jnp.asarray(X, float), jnp.asarray(y, float)
        d = X.shape[1]
        th0 = np.r_[np.full(d, np.log(self.init[0])), np.log(self.init[1]), np.log(self.init[2])]
        f = jax.jit(jax.value_and_grad(lambda t: _nlml(t, X, y, d)))

        def fun(t):
            v, g = f(jnp.asarray(t))
            return float(v), np.asarray(g, dtype=float)

        bounds = [(np.log(1e-2), np.log(1e5))] * d + [(np.log(1e-3), np.log(1e4)), (np.log(1e-4), np.log(1e2))]
        res = minimize(fun, th0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.maxiter})
        self.theta_, self.nlml_ = res.x, res.fun
        self.X_, self.y_, self.d_ = X, y, d
        logl, logsf, logsn = res.x[:d], res.x[d], res.x[d + 1]
        K = _k(X, X, logl, logsf) + (np.exp(2 * logsn) + 1e-8) * jnp.eye(X.shape[0])
        self.L_ = jnp.linalg.cholesky(K)
        self.alpha_ = jax.scipy.linalg.cho_solve((self.L_, True), y)
        return self

    @property
    def length_scales(self):
        return np.exp(self.theta_[: self.d_])

    @property
    def noise_std(self):
        return float(np.exp(self.theta_[self.d_ + 1]))

    def predict(self, Xs, return_std=False):
        Xs = jnp.asarray(Xs, float)
        d = self.d_
        logl, logsf, logsn = self.theta_[:d], self.theta_[d], self.theta_[d + 1]
        Ks = _k(Xs, self.X_, logl, logsf)
        mu = Ks @ self.alpha_
        if not return_std:
            return np.asarray(mu)
        v = jax.scipy.linalg.solve_triangular(self.L_, Ks.T, lower=True)
        var = np.exp(2 * logsf) - jnp.sum(v**2, 0) + np.exp(2 * logsn)  # incl. noise, like GPML ys2
        return np.asarray(mu), np.sqrt(np.maximum(np.asarray(var), 0))

    def ard_ranking(self):
        """Feature indices sorted from most to least relevant (shortest length scale first)."""
        return np.argsort(self.length_scales)

    def ard_weights(self):
        """w_m = exp(-l_m) normalised to sum 1 (paper Methods; authors' ARD_GPR.m)."""
        w = np.exp(-self.length_scales)
        return w / w.sum()


def fit_ard_restarts(X, y, n_restarts=8, seed=0, maxiter=300):
    """ARD likelihood is multi-modal (neighbouring frequencies are almost collinear).
    Fit from the paper's start (all l = 1, sf = 1, sn = 0.1) plus random starts
    (log l ~ U[log 0.3, log 10], sf = rms(y) or 1, centred or raw y) and keep every fit.
    Returns (best_model, list_of_models); 'best' = lowest negative log marginal likelihood.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    fits = []
    for r in range(n_restarts):
        m = ARDGP(maxiter=maxiter)
        if r == 0:
            th_l = np.zeros(X.shape[1])
            sf = 1.0
        else:
            th_l = rng.uniform(np.log(0.3), np.log(10.0), X.shape[1])
            sf = float(np.sqrt(np.mean(y**2))) if r % 2 else 1.0
        m.init = (1.0, sf, 0.1)
        m._th_l = th_l
        fits.append(_fit_from(m, X, y, th_l, sf))
    ok = [m for m in fits if np.isfinite(m.nlml_) and np.isfinite(m.ard_weights()).all()]
    best = min(ok, key=lambda m: m.nlml_)
    return best, ok


def _fit_from(m, X, y, th_l, sf):
    Xj, yj = jnp.asarray(X, float), jnp.asarray(y, float)
    d = Xj.shape[1]
    th0 = np.r_[th_l, np.log(sf), np.log(0.1)]
    f = jax.jit(jax.value_and_grad(lambda t: _nlml(t, Xj, yj, d)))

    def fun(t):
        v, g = f(jnp.asarray(t))
        return float(v), np.asarray(g, dtype=float)

    bounds = [(np.log(1e-2), np.log(1e5))] * d + [(np.log(1e-3), np.log(1e4)), (np.log(1e-4), np.log(1e2))]
    res = minimize(fun, th0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": m.maxiter})
    m.theta_, m.nlml_, m.X_, m.y_, m.d_ = res.x, res.fun, Xj, yj, d
    logl, logsf, logsn = res.x[:d], res.x[d], res.x[d + 1]
    K = _k(Xj, Xj, logl, logsf) + (np.exp(2 * logsn) + 1e-8) * jnp.eye(Xj.shape[0])
    m.L_ = jnp.linalg.cholesky(K)
    m.alpha_ = jax.scipy.linalg.cho_solve((m.L_, True), yj)
    return m
