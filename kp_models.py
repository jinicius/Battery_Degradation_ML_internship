"""Models, features and evaluation helpers for the LMB knee-point (KP) study.

Target, as in the paper's code: y = n_cycle / KP = 10 / KP  (a 'rate' that grows when the knee comes early).
Every model predicts y; KP_hat = 10 / y_hat. Errors are reported on KP (MAPE %, MAE in cycles).
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import GroupKFold

N_SEG = 10            # DoD segments of 10 % each (20 points of the 201-point grid)


# ------------------------------------------------------------------ targets and metrics
def to_y(kp, n_cycle=10):
    return n_cycle / np.asarray(kp, float)


def to_kp(y, n_cycle=10):
    return n_cycle / np.clip(np.asarray(y, float), 0.05, None)


def mape(kp, kp_hat):
    kp, kp_hat = np.asarray(kp, float), np.asarray(kp_hat, float)
    return float(np.mean(np.abs(kp_hat - kp) / kp) * 100)


def mae(kp, kp_hat):
    return float(np.mean(np.abs(np.asarray(kp_hat) - np.asarray(kp))))


# ------------------------------------------------------------------ features
def seg_index(n_pt=201, n_seg=N_SEG):
    edges = np.linspace(0, n_pt - 1, n_seg + 1).round().astype(int)
    return [np.arange(edges[s], edges[s + 1] + (1 if s == n_seg - 1 else 0)) for s in range(n_seg)]


def area_variation(X, n_cycle=10, segs=None):
    """AV_s = mean_{j=1..n-1} sum_{i in segment s} |V_{i,j} - V_{i,0}|   (paper's get_area_var)
    X: (n_cell, 201, >=n_cycle) raw voltages. Returns (n_cell, n_seg)."""
    segs = seg_index(X.shape[1]) if segs is None else segs
    dV = np.abs(X[:, :, 1:n_cycle] - X[:, :, :1])            # (n, 201, n-1)
    return np.stack([dV[:, s, :].sum(1).mean(1) for s in segs], axis=1)


def seg_features(X, n_cycle=10):
    """20 engineered features per cell: log10 AV_s (10 segments) + mean cycle-0 voltage per segment."""
    segs = seg_index(X.shape[1])
    av = np.log10(area_variation(X, n_cycle, segs) + 1e-6)
    v0 = np.stack([X[:, s, 0].mean(1) for s in segs], axis=1)
    return np.hstack([av, v0])


SEG_FEATURE_SEGMENT = np.r_[np.arange(N_SEG), np.arange(N_SEG)]   # which DoD segment each seg-feature belongs to


def raw_features(Xs, n_cycle=10):
    """The paper's 2-D input for non-NN models: [V_11..V_m1, V_12..V_m10] (scaled voltages)."""
    return Xs[:, :, :n_cycle].transpose(0, 2, 1).reshape(len(Xs), -1)


# ------------------------------------------------------------------ splits
def random_splits(n, repeats=20, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        p = rng.permutation(n)
        k = int(round(n * test_frac))
        out.append((np.sort(p[k:]), np.sort(p[:k])))
    return out


def group_splits(groups, n_splits=5):
    idx = np.arange(len(groups))
    return [(tr, te) for tr, te in GroupKFold(n_splits).split(idx, groups=groups)]


def chem_splits(chem):
    idx = np.arange(len(chem))
    return [(idx[chem == "NMC"], idx[chem == "S"]), (idx[chem == "S"], idx[chem == "NMC"])]


# ------------------------------------------------------------------ the paper's 1D CNN in JAX
# Conv1D(4 filters, kernel 10, valid, ReLU) -> Dropout(0.2) -> MaxPool(5) -> Flatten -> Dense(20, ReLU) -> Dense(1)
DN = ("NWC", "WIO", "NWC")


def cnn_init(key, n_t=201, n_c=10, f=4, k=10, pool=5, hidden=20):
    k1, k2, k3 = jax.random.split(key, 3)
    L = n_t - k + 1
    P = L // pool
    glorot = lambda kk, shape, fan_in, fan_out: jax.random.uniform(
        kk, shape, minval=-np.sqrt(6 / (fan_in + fan_out)), maxval=np.sqrt(6 / (fan_in + fan_out)))
    return {"W": glorot(k1, (k, n_c, f), k * n_c, k * f), "b": jnp.zeros(f),
            "W1": glorot(k2, (P * f, hidden), P * f, hidden), "b1": jnp.zeros(hidden),
            "W2": glorot(k3, (hidden, 1), hidden, 1), "b2": jnp.zeros(1),
            "_meta": jnp.array([k, pool])}


def cnn_conv(p, x):
    """A = ReLU(conv(x)) : (B, L, f) with L = 201 - 10 + 1 = 192."""
    return jax.nn.relu(jax.lax.conv_general_dilated(x, p["W"], (1,), "VALID", dimension_numbers=DN) + p["b"])


def cnn_head(p, A, pool=5):
    B, L, f = A.shape
    P = L // pool
    h = A[:, :P * pool].reshape(B, P, pool, f).max(2).reshape(B, -1)
    h = jax.nn.relu(h @ p["W1"] + p["b1"])
    return (h @ p["W2"] + p["b2"])[:, 0]


def cnn_forward(p, x, key=None, drop=0.2):
    A = cnn_conv(p, x)
    if key is not None:                               # dropout only while training
        keep = jax.random.bernoulli(key, 1 - drop, A.shape)
        A = A * keep / (1 - drop)
    return cnn_head(p, A)


def cnn_train(Xtr, ytr, seed=0, epochs=500, batch=8, lr=1e-3, val_frac=0.1, loss="mae"):
    """Adam on MAE; 10 % of the training cells held out; keep the epoch with the lowest validation loss."""
    rng = np.random.default_rng(seed)
    n = len(ytr)
    perm = rng.permutation(n)
    nv = max(1, int(round(n * val_frac)))
    iv, it = perm[:nv], perm[nv:]
    key = jax.random.PRNGKey(seed)
    key, sk = jax.random.split(key)
    p = cnn_init(sk, n_t=Xtr.shape[1], n_c=Xtr.shape[2])
    meta = p.pop("_meta")
    opt = optax.adam(lr)
    st = opt.init(p)
    lossf = (lambda a, b: jnp.mean(jnp.abs(a - b))) if loss == "mae" else (lambda a, b: jnp.mean((a - b) ** 2))

    @jax.jit
    def step(p, st, x, y, k):
        l, g = jax.value_and_grad(lambda p: lossf(cnn_forward(p, x, k), y))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    @jax.jit
    def vloss(p, x, y):
        return lossf(cnn_forward(p, x), y)

    Xt, yt, Xv, yv = jnp.asarray(Xtr[it]), jnp.asarray(ytr[it]), jnp.asarray(Xtr[iv]), jnp.asarray(ytr[iv])
    best, best_p = np.inf, p
    for ep in range(epochs):
        o = rng.permutation(len(it))
        for i in range(0, len(o), batch):
            b = o[i:i + batch]
            key, sk = jax.random.split(key)
            p, st, _ = step(p, st, Xt[b], yt[b], sk)
        v = float(vloss(p, Xv, yv))
        if v < best:
            best, best_p = v, p
    return best_p


def cnn_predict(p, X):
    return np.asarray(cnn_forward(p, jnp.asarray(X)))


def gradcam(p, X, n_t=201, k=10):
    """Grad-CAM on the conv layer (as tf-keras-vis with penultimate_layer = conv):
       alpha_k = mean_t dy/dA_{t,k};  cam_t = ReLU(sum_k alpha_k A_{t,k});  min-max to [0,1];
       conv position t covers input points t..t+9 -> placed at its centre and interpolated to 201 points."""
    X = jnp.asarray(X)
    A = cnn_conv(p, X)
    G = jax.grad(lambda A_: cnn_head(p, A_).sum())(A)
    alpha = G.mean(1)                                    # (B, f)
    cam = jax.nn.relu((A * alpha[:, None, :]).sum(-1))   # (B, L)
    cam = np.asarray(cam)
    lo, hi = cam.min(1, keepdims=True), cam.max(1, keepdims=True)
    cam = (cam - lo) / np.where(hi - lo > 0, hi - lo, 1)
    centres = np.arange(cam.shape[1]) + (k - 1) / 2
    grid = np.arange(n_t)
    return np.stack([np.interp(grid, centres, c) for c in cam])


def vanilla_saliency(p, X):
    """|dy/dx| summed over the 10 cycle channels -> (B, 201)."""
    g = jax.vmap(jax.grad(lambda x: cnn_forward(p, x[None])[0]))(jnp.asarray(X))
    return np.abs(np.asarray(g)).sum(-1)


# ------------------------------------------------------------------ inter-cell CNN (BatLiNet idea)
def cnn_train_inter(Xtr, ytr, seed=0, epochs=300, pairs=256, lr=1e-3):
    """Learn dy = g(x_i - x_j) with the same CNN body; random pairs every step."""
    key = jax.random.PRNGKey(1000 + seed)
    key, sk = jax.random.split(key)
    p = cnn_init(sk, n_t=Xtr.shape[1], n_c=Xtr.shape[2])
    p.pop("_meta")
    opt = optax.adam(lr)
    st = opt.init(p)
    Xj, Yj = jnp.asarray(Xtr), jnp.asarray(ytr)

    @jax.jit
    def step(p, st, i, j, k):
        def l(p):
            return jnp.mean(jnp.abs(cnn_forward(p, Xj[i] - Xj[j], k) - (Yj[i] - Yj[j])))
        v, g = jax.value_and_grad(l)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, v

    rng = np.random.default_rng(seed)
    n = len(ytr)
    for ep in range(epochs):
        i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
        key, sk = jax.random.split(key)
        p, st, _ = step(p, st, jnp.asarray(i), jnp.asarray(j), sk)
    return p


def inter_predict(p, Xte, Xref, yref):
    """y_hat(x) = mean_r [ g(x - x_r) + y_r ]."""
    out = []
    for x in Xte:
        d = jnp.asarray(x[None] - Xref)
        out.append(float(np.mean(np.asarray(cnn_forward(p, d)) + yref)))
    return np.array(out)
