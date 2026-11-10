"""Small 1D CNN for EIS spectra in JAX + gradient saliency maps.

Following Liu et al., JACS 146, 33012 (2024) ("Interpretable learning of accelerated aging in
lithium metal batteries"): a 1D CNN reads a curve end-to-end and a saliency map, the gradient of
the prediction with respect to each input point, shows which part of the curve the model relies
on. Here the curve is the EIS spectrum: 60 frequencies x 2 channels (Re Z, -Im Z).
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

DN = ("NWC", "WIO", "NWC")


def init_params(key, ch=(2, 16, 32), k=5, n_freq=60, hidden=64):
    ks = jax.random.split(key, 4)
    p = {}
    p["c1"] = jax.random.normal(ks[0], (k, ch[0], ch[1])) * np.sqrt(2 / (k * ch[0]))
    p["b1"] = jnp.zeros(ch[1])
    p["c2"] = jax.random.normal(ks[1], (k, ch[1], ch[2])) * np.sqrt(2 / (k * ch[1]))
    p["b2"] = jnp.zeros(ch[2])
    flat = (n_freq // 4) * ch[2]
    p["w3"] = jax.random.normal(ks[2], (flat, hidden)) * np.sqrt(2 / flat)
    p["b3"] = jnp.zeros(hidden)
    p["w4"] = jax.random.normal(ks[3], (hidden, 1)) * np.sqrt(1 / hidden)
    p["b4"] = jnp.zeros(1)
    return p


def _pool(h):
    B, W, C = h.shape
    return h[:, : W // 2 * 2].reshape(B, W // 2, 2, C).mean(2)


def forward(p, x):
    """x: (B, 60, 2) standardised spectrum -> (B,) standardised target."""
    h = jax.lax.conv_general_dilated(x, p["c1"], (1,), "SAME", dimension_numbers=DN) + p["b1"]
    h = _pool(jax.nn.gelu(h))
    h = jax.lax.conv_general_dilated(h, p["c2"], (1,), "SAME", dimension_numbers=DN) + p["b2"]
    h = _pool(jax.nn.gelu(h))
    h = h.reshape(h.shape[0], -1)
    h = jax.nn.gelu(h @ p["w3"] + p["b3"])
    return (h @ p["w4"] + p["b4"])[:, 0]


def to_seq(Z):
    """(n, 120) [Re(60), Im(60)] -> (n, 60, 2), frequency order 20 kHz -> 0.02 Hz."""
    return np.stack([Z[:, :60], Z[:, 60:]], axis=-1)


def train(Xs, ys, Xv=None, yv=None, seed=0, epochs=400, lr=2e-3, wd=1e-4, batch=64, noise=0.02):
    key = jax.random.PRNGKey(seed)
    p = init_params(key)
    opt = optax.adamw(lr, weight_decay=wd)
    st = opt.init(p)

    def loss(p, x, y):
        return jnp.mean((forward(p, x) - y) ** 2)

    @jax.jit
    def step(p, st, x, y, k):
        x = x + noise * jax.random.normal(k, x.shape)       # small input noise = augmentation
        l, g = jax.value_and_grad(loss)(p, x, y)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    Xs, ys = jnp.asarray(Xs), jnp.asarray(ys)
    n = len(ys)
    rng = np.random.default_rng(seed)
    best, best_p, hist = np.inf, p, []
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            key, sk = jax.random.split(key)
            p, st, l = step(p, st, Xs[idx], ys[idx], sk)
        if Xv is not None and (ep % 10 == 9 or ep == epochs - 1):
            v = float(loss(p, jnp.asarray(Xv), jnp.asarray(yv)))
            hist.append(v)
            if v < best:
                best, best_p = v, p
    return (best_p if Xv is not None else p), hist


def predict(p, X):
    return np.asarray(forward(p, jnp.asarray(X)))


def saliency(p, X):
    """d prediction / d input for every sample: (n, 60, 2)."""
    g = jax.vmap(jax.grad(lambda x: forward(p, x[None])[0]))
    return np.asarray(g(jnp.asarray(X)))
