import numpy as np
from panel import Panel
from apparatus import (Apparatus, FeatureMap, boltzmann_advisor, LinearSoftmaxPolicy, sequenced_component_update)
from rollout import (collect_rollouts, empirical_occupancy, compute_gae, fit_value_tabular, bucket_by_sa)
from sampled_apparatus import SampledApparatus
from recording import save_run
from exp_control import (build, BETA0, TAU, LR, DELTA_C, DELTA_PHI, D_FEAT, DBUDGET, J_OPT)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


def _torch_device():
    return "cuda" if (_TORCH and torch.cuda.is_available()) else "cpu"

GAE_LAMBDA = 0.95

BETA0_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
GATE_BAND = (0.2, 0.5)
STRESS_BAND = (0.6, 0.8)

def _band(beta0):
    if GATE_BAND[0] - 1e-9 <= beta0 <= GATE_BAND[1] + 1e-9:
        return "gate"
    if STRESS_BAND[0] - 1e-9 <= beta0 <= STRESS_BAND[1] + 1e-9:
        return "stress"
    return "?"

def _independent_solve(P, R, gamma, mu0, pi):
    S, A, _ = P.shape
    dt = np.result_type(np.asarray(pi).dtype, np.float64)
    P = P.astype(dt, copy=False)
    R = R.astype(dt, copy=False)
    mu0 = mu0.astype(dt, copy=False)
    I = np.eye(S, dtype=dt)
    P_pi = np.einsum("sa,sax->sx", pi, P)
    d = np.linalg.solve((I - gamma * P_pi).T, (1.0 - gamma) * mu0)
    R_pi = np.einsum("sa,sa->s", pi, R)
    V = np.linalg.solve(I - gamma * P_pi, R_pi)
    Q = R + gamma * np.einsum("sax,x->sa", P, V)
    A = Q - (pi * Q).sum(axis=1, keepdims=True)
    J = mu0 @ V
    return d, V, Q, A, J


def _mix(pi_A, pi_B, beta):
    return (1.0 - beta) * pi_A + beta * pi_B


def oracle_beta_dial(panel, env, pi_A, pi_B, beta, *, eps=1e-3):
    P, R_ext, g, mu0 = env.P, env.R_ext, env.gamma, env.mu0
    gr = panel.control_gradient(pi_A, pi_B, beta, tau=None)

    d, _, _, AE, _ = _independent_solve(P, R_ext, g, mu0, _mix(pi_A, pi_B, beta))
    G_indep = float((d @ ((pi_A - pi_B) * AE).sum(axis=1)).real)
    dJdb_indep = -G_indep / (1.0 - g)

    h = 1e-30
    _, _, _, _, J_c = _independent_solve(P, R_ext, g, mu0, _mix(pi_A, pi_B, beta + 1j * h))
    dJdb_cs = float(J_c.imag / h)

    _, _, _, _, Jp = _independent_solve(P, R_ext, g, mu0, _mix(pi_A, pi_B, beta + eps))
    _, _, _, _, Jm = _independent_solve(P, R_ext, g, mu0, _mix(pi_A, pi_B, beta - eps))
    dJdb_fd = float((Jp - Jm).real / (2 * eps))

    return dict(
        panel_G=gr.G_beta, indep_G=G_indep, err_G=abs(gr.G_beta - G_indep),
        panel_dJdb=gr.dJ_dbeta,
        err_construction=abs(gr.dJ_dbeta - dJdb_indep),
        err_formula_cs=abs(gr.dJ_dbeta - dJdb_cs),
        err_formula_fd=abs(gr.dJ_dbeta - dJdb_fd),
    )


def run_oracle(panel, env, configs, *, tol=1e-9, verbose=True):
    if verbose:
        print("=== ORACLE: panel beta-dial certification (no panel code on the check side) ===")
        print(" config                       | err_construct | err_formula_cs | err_fd(eps=1e-3)")
    ok = True
    for name, pi_A, pi_B, beta in configs:
        r = oracle_beta_dial(panel, env, pi_A, pi_B, beta)
        passed = (r["err_construction"] < tol) and (r["err_formula_cs"] < tol)
        ok = ok and passed
        if verbose:
            print(f" {name:28s} |   {r['err_construction']:.2e}    |   "
                  f"{r['err_formula_cs']:.2e}    |   {r['err_formula_fd']:.2e}   "
                  f"{'OK' if passed else 'FAIL'}")
    if verbose:
        print(f"  -> oracle {'PASSED' if ok else 'FAILED'} "
              f"(construction + complex-step both < {tol:g}); real-FD is O(eps^2) truncation, informative\n")
    return ok


class UVFAQCritic:
    def __init__(self, S, A, m, beta_grid, *, seed=0, beta_basis="rbf",
                 k_beta=4, ridge=1e-6, features="random"):
        self.S, self.A = int(S), int(A)
        rng = np.random.default_rng(seed)
        if features == "tabular" or m >= S:
            self.Psi = np.eye(S); self.m = S
        elif features == "rbf":
            size = int(round(S ** 0.5)); gg = max(int(round(m ** 0.5)), 1)
            rows = (np.arange(S) // size) / max(size - 1, 1)
            cols = (np.arange(S) % size) / max(size - 1, 1)
            cy, cx = np.meshgrid(np.linspace(0, 1, gg), np.linspace(0, 1, gg))
            centers = np.stack([cy.ravel(), cx.ravel()], 1)
            wid = (1.0 / (gg - 1)) if gg > 1 else 1.0
            pos = np.stack([rows, cols], 1)
            d2 = ((pos[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            self.Psi = np.exp(-0.5 * d2 / (wid ** 2)); self.m = self.Psi.shape[1]
        else:
            self.Psi = rng.standard_normal((S, m)) / np.sqrt(m); self.m = int(m)
        self.tabular = (features == "tabular" or m >= S)
        self.beta_basis = beta_basis
        self.k_beta = int(k_beta)
        lo, hi = float(min(beta_grid)), float(max(beta_grid))
        if beta_basis == "rbf":
            self.centers = np.linspace(lo, hi, self.k_beta)
            self.width = (hi - lo) / max(self.k_beta - 1, 1) + 1e-9
        self.ridge = float(ridge)
        self.U = None

    def _phi_beta(self, beta_vals):
        b = np.atleast_1d(np.asarray(beta_vals, dtype=float))
        if self.beta_basis == "rbf":
            return np.exp(-0.5 * ((b[:, None] - self.centers[None, :]) / self.width) ** 2)
        return np.vander(b, self.k_beta, increasing=True)

    def _features(self, s_idx, beta_vals):
        Ps = self.Psi[np.asarray(s_idx, dtype=int)]
        Pb = self._phi_beta(beta_vals)
        return (Ps[:, :, None] * Pb[:, None, :]).reshape(len(Ps), -1)

    def fit(self, s_idx, a_idx, beta_vals, targets):
        X = self._features(s_idx, beta_vals)
        Pdim = X.shape[1]
        a_idx = np.asarray(a_idx, dtype=int)
        targets = np.asarray(targets, dtype=float)
        self.U = np.zeros((self.A, Pdim))
        ridgeI = self.ridge * np.eye(Pdim)
        for a in range(self.A):
            mask = a_idx == a
            if not mask.any():
                continue
            Xa = X[mask]
            self.U[a] = np.linalg.solve(Xa.T @ Xa + ridgeI, Xa.T @ targets[mask])

    def Q(self, beta):
        s_all = np.arange(self.S)
        X = self._features(s_all, np.full(self.S, beta))
        return X @ self.U.T

    def advantage(self, beta, pi_phi):
        Q = self.Q(beta)
        return Q - (pi_phi * Q).sum(axis=1, keepdims=True)


class MLPQCritic:
    def __init__(self, env, A, beta_grid, *, hidden=(128, 128), k_beta=4, seed=0,
                 lr=3e-3, epochs=200, batch=512, l2=1e-6, max_samples=40000,
                 input_mode="onehot"):
        self.env, self.A = env, int(A)
        self.size = int(round(env.n_states ** 0.5)); self.input_mode = input_mode
        self.rng = np.random.default_rng(seed)
        lo, hi = float(min(beta_grid)), float(max(beta_grid))
        self.centers = np.linspace(lo, hi, k_beta); self.width = (hi - lo) / max(k_beta - 1, 1) + 1e-9
        din = (env.n_states if input_mode == "onehot" else 2) + k_beta
        dims = [din] + list(hidden) + [self.A]
        self.W = [self.rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.lr, self.epochs, self.batch, self.l2, self.max_samples = lr, epochs, batch, l2, max_samples

    def _feat(self, s_idx, beta_vals):
        s = np.asarray(s_idx); b = np.atleast_1d(np.asarray(beta_vals, float))
        if b.shape[0] == 1 and len(s) > 1:
            b = np.full(len(s), b[0])
        rbf = np.exp(-0.5 * ((b[:, None] - self.centers[None, :]) / self.width) ** 2)
        if self.input_mode == "onehot":
            oh = np.zeros((len(s), self.env.n_states)); oh[np.arange(len(s)), s] = 1.0
            return np.concatenate([oh, rbf], 1)
        r = (s // self.size) / (self.size - 1); c = (s % self.size) / (self.size - 1)
        return np.concatenate([np.stack([r, c], 1), rbf], 1)

    def _fwd(self, X, cache=False):
        a = X; cs = []
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            if cache:
                cs.append((a, z))
            a = np.maximum(z, 0.0)
        out = a @ self.W[-1] + self.b[-1]
        if cache:
            cs.append((a, None))
        return (out, cs) if cache else out

    def fit(self, s_idx, a_idx, beta_vals, targets):
        X = self._feat(s_idx, beta_vals); a_idx = np.asarray(a_idx)
        y = np.asarray(targets, float); N = len(y)
        if N > self.max_samples:
            k = self.rng.choice(N, self.max_samples, replace=False)
            X, a_idx, y, N = X[k], a_idx[k], y[k], self.max_samples
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        t = 0; b1, b2, eps = 0.9, 0.999, 1e-8
        for _ in range(self.epochs):
            idx = self.rng.permutation(N)
            for st in range(0, N, self.batch):
                bi = idx[st:st + self.batch]; Xb = X[bi]; ab = a_idx[bi]; yb = y[bi]; B = len(bi)
                out, cs = self._fwd(Xb, cache=True); pred = out[np.arange(B), ab]
                dout = np.zeros_like(out); dout[np.arange(B), ab] = 2.0 * (pred - yb) / B
                gW = [None] * len(self.W); gb = [None] * len(self.b); da = dout
                gW[-1] = cs[-1][0].T @ da + self.l2 * self.W[-1]; gb[-1] = da.sum(0); da = da @ self.W[-1].T
                for i in range(len(self.W) - 2, -1, -1):
                    a_prev, z = cs[i]; da = da * (z > 0)
                    gW[i] = a_prev.T @ da + self.l2 * self.W[i]; gb[i] = da.sum(0); da = da @ self.W[i].T
                t += 1
                for P, gg, mm, vv in [(self.W, gW, mW, vW), (self.b, gb, mb, vb)]:
                    for i in range(len(P)):
                        mm[i] = b1 * mm[i] + (1 - b1) * gg[i]; vv[i] = b2 * vv[i] + (1 - b2) * gg[i] ** 2
                        P[i] -= self.lr * (mm[i] / (1 - b1 ** t)) / (np.sqrt(vv[i] / (1 - b2 ** t)) + eps)
        return self

    def Q(self, beta):
        s = np.arange(self.env.n_states)
        return self._fwd(self._feat(s, np.full(self.env.n_states, beta)))

    def advantage(self, beta, pi_phi):
        Q = self.Q(beta)
        return Q - (pi_phi * Q).sum(axis=1, keepdims=True)


def build_local_view(env, crop=7):
    S, size = env.n_states, env.size
    R = crop // 2
    DR = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    wset = set()
    for a, b in env.wall_edges:
        wset.add((a, b)); wset.add((b, a))
    wall = np.zeros((S, 4), np.float32)
    for s in range(S):
        r, c = env._to_cell(s)
        for d, (dr, dc) in enumerate(DR):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < size and 0 <= nc < size) or ((r, c), (nr, nc)) in wset:
                wall[s, d] = 1.0
    obs = np.zeros((S, 7, crop, crop), np.float32)
    for s in range(S):
        r, c = env._to_cell(s)
        for i, rr in enumerate(range(r - R, r + R + 1)):
            for j, cc in enumerate(range(c - R, c + R + 1)):
                if not (0 <= rr < size and 0 <= cc < size):
                    obs[s, 5, i, j] = 1.0; obs[s, 0:4, i, j] = 1.0
                    continue
                t = env._to_index((rr, cc))
                obs[s, 0:4, i, j] = wall[t]
                if t == env._goal_s:
                    obs[s, 6, i, j] = 1.0
        obs[s, 4, R, R] = 1.0
    return obs


class _TorchQCritic:
    def __init__(self, env, A, beta_grid, *, mode, seed=0, k_beta=4, lr=3e-3, epochs=200, batch=512, hidden=128, crop=7, device=None):
        if not _TORCH:
            raise RuntimeError("torch unavailable; route mlp_* to numpy MLPQCritic fallback")
        self.env, self.A, self.mode = env, int(A), mode
        self.device = torch.device(device or _torch_device())
        torch.manual_seed(seed)
        lo, hi = float(min(beta_grid)), float(max(beta_grid))
        self.centers = torch.tensor(np.linspace(lo, hi, k_beta), dtype=torch.float32, device=self.device)
        self.width = float((hi - lo) / max(k_beta - 1, 1) + 1e-9)
        self.k_beta, self.lr, self.epochs, self.batch = k_beta, lr, epochs, batch
        S = env.n_states
        if mode == "onehot":
            X = np.eye(S, dtype=np.float32); enc_out = S; self.enc = nn.Identity()
        elif mode == "geom":
            r = (np.arange(S) // env.size) / (env.size - 1); c = (np.arange(S) % env.size) / (env.size - 1)
            X = np.stack([r, c], 1).astype(np.float32); enc_out = 2; self.enc = nn.Identity()
        elif mode == "cnn_local":
            X = build_local_view(env, crop); ch = X.shape[1]
            self.enc = nn.Sequential(nn.Conv2d(ch, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten())
            enc_out = 32 * crop * crop
        else:
            raise ValueError(mode)
        self.Xt = torch.tensor(X, device=self.device)
        self.head = nn.Sequential(nn.Linear(enc_out + k_beta, hidden), nn.ReLU(), nn.Linear(hidden, self.A))
        self.enc.to(self.device)
        self.head.to(self.device)

    def _rbf(self, beta_t):
        return torch.exp(-0.5 * ((beta_t[:, None] - self.centers[None, :]) / self.width) ** 2)

    def fit(self, s_idx, a_idx, beta_vals, targets):
        s = torch.as_tensor(np.asarray(s_idx), dtype=torch.long, device=self.device)
        a = torch.as_tensor(np.asarray(a_idx), dtype=torch.long, device=self.device)
        bv = np.atleast_1d(np.asarray(beta_vals, np.float32))
        if bv.shape[0] == 1 and len(s) > 1:
            bv = np.full(len(s), bv[0], np.float32)
        b = torch.as_tensor(bv, device=self.device)
        y = torch.as_tensor(np.asarray(targets, np.float32), device=self.device)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.head.parameters()), lr=self.lr)
        N = len(s)
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            for st in range(0, N, self.batch):
                bi = perm[st:st + self.batch]
                h = self.enc(self.Xt[s[bi]])
                out = self.head(torch.cat([h, self._rbf(b[bi])], 1))
                pred = out.gather(1, a[bi][:, None]).squeeze(1)
                loss = ((pred - y[bi]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def Q(self, beta):
        with torch.no_grad():
            s = torch.arange(self.env.n_states, device=self.device)
            h = self.enc(self.Xt[s])
            b = torch.full((self.env.n_states,), float(beta), device=self.device)
            return self.head(torch.cat([h, self._rbf(b)], 1)).cpu().numpy()

    def advantage(self, beta, pi_phi):
        Q = self.Q(beta)
        return Q - (pi_phi * Q).sum(axis=1, keepdims=True)


def build_critic(spec, env, beta_grid, *, seed=0, mlp_epochs=200, prefer_torch=True):
    S, A = env.n_states, env.n_actions
    if spec == "tabular":
        return UVFAQCritic(S, A, S, beta_grid, seed=seed, features="tabular")
    if spec.startswith("rbf"):
        return UVFAQCritic(S, A, int(spec[3:]), beta_grid, seed=seed, features="rbf")
    if spec.startswith("rand"):
        return UVFAQCritic(S, A, int(spec[4:]), beta_grid, seed=seed, features="random")
    if spec == "cnn_local":
        if not _TORCH:
            raise RuntimeError("cnn_local requires torch (run on the GPU box)")
        return _TorchQCritic(env, A, beta_grid, mode="cnn_local", seed=seed, epochs=mlp_epochs)
    if spec in ("mlp_onehot", "mlp_geom"):
        mode = "onehot" if spec == "mlp_onehot" else "geom"
        if prefer_torch and _TORCH:
            return _TorchQCritic(env, A, beta_grid, mode=mode, seed=seed, epochs=mlp_epochs)
        return MLPQCritic(env, A, beta_grid, seed=seed, input_mode=mode, epochs=mlp_epochs)
    raise ValueError(f"unknown critic spec {spec!r}")


def collect_critic_samples(env, pi_A, pi_B, beta_grid, stream, *, n_episodes, gae_lambda, critic_iters, seed):
    g = env.gamma
    rng = np.random.default_rng(seed)
    S_, A_, B_, Y_ = [], [], [], []
    for beta in beta_grid:
        pi_phi = _mix(pi_A, pi_B, beta)
        roll = collect_rollouts(env, pi_phi, n_episodes, seed=int(rng.integers(1 << 31)))
        V = fit_value_tabular(roll, stream, g, env.n_states,
                              gae_lambda=gae_lambda, n_iters=critic_iters)
        _, value_target = compute_gae(roll, V, stream, g, gae_lambda)
        S_.append(roll.s); A_.append(roll.a)
        B_.append(np.full(roll.s.shape, beta, dtype=float)); Y_.append(value_target)
    return (np.concatenate(S_), np.concatenate(A_),
            np.concatenate(B_), np.concatenate(Y_))


_PHI_BINS = [0.0, 1e-3, 1e-2, 1e-1, 1.0 + 1e-9]
_PHI_LABELS = ["[0,1e-3)", "[1e-3,1e-2)", "[1e-2,0.1)", "[0.1,1]"]


def occ_weighted_fidelity(panel, critic, pi_A, pi_B, beta, stream="ext"):
    pi_phi = _mix(pi_A, pi_B, beta)
    sol = panel.solve(pi_phi)
    A_exact = (sol.A_ext if stream == "ext" else sol.A_int).ravel()
    A_hat = critic.advantage(beta, pi_phi).ravel()
    w = (sol.d[:, None] * pi_phi).ravel()
    xm, ym = (w * A_hat).sum() / w.sum(), (w * A_exact).sum() / w.sum()
    cov = (w * (A_hat - xm) * (A_exact - ym)).sum()
    sx = np.sqrt((w * (A_hat - xm) ** 2).sum()); sy = np.sqrt((w * (A_exact - ym) ** 2).sum())
    corr = float(cov / (sx * sy + 1e-18))
    mag = float((w * np.abs(A_hat)).sum() / (w * np.abs(A_exact)).sum())
    return corr, mag, float((w * np.abs(A_exact)).sum())


def stratified_adv_error(panel, critic, pi_A, pi_B, beta, stream="ext"):
    pi_phi = _mix(pi_A, pi_B, beta)
    sol = panel.solve(pi_phi)
    A_exact = sol.A_ext if stream == "ext" else sol.A_int
    A_hat = critic.advantage(beta, pi_phi)
    err = np.abs(A_hat - A_exact)
    binidx = np.digitize(pi_phi.ravel(), _PHI_BINS) - 1
    rows = []
    for b, lab in enumerate(_PHI_LABELS):
        m = binidx == b
        cnt = int(m.sum())
        if cnt:
            rows.append((lab, cnt, float(err.ravel()[m].mean()), float(err.ravel()[m].max())))
        else:
            rows.append((lab, 0, float("nan"), float("nan")))
    return rows, pi_phi, A_exact, A_hat


def sampled_beta_dial(env, panel, critic, pi_A, pi_B, beta, *, n_episodes, seed):
    pi_phi = _mix(pi_A, pi_B, beta)
    roll = collect_rollouts(env, pi_phi, n_episodes, seed=seed)
    d_hat = empirical_occupancy(roll, env.n_states, env.gamma)
    A_hat = critic.advantage(beta, pi_phi)
    G_hat = float(d_hat @ ((pi_A - pi_B) * A_hat).sum(axis=1))
    G_exact = panel.control_gradient(pi_A, pi_B, beta, tau=None).G_beta
    return G_exact, G_hat

def smoke(seed=0, warm_iters=250, n_episodes=150, critic_iters=20, beta_grid=(0.10, 0.30, 0.45), eval_beta=0.45, m_list=(16, "tabular")):
    env, panel, room_states = build()
    S, A = env.n_states, env.n_actions
    beta_grid = list(beta_grid)

    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    ap = Apparatus(panel, fm, A, advisor_mode="shaped", beta=BETA0, lam=1.0,
                   tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                   distractor_states=room_states, seed=seed)
    for _ in range(warm_iters):
        ap.step(beta=BETA0, lam=1.0)
    pi_A, pi_B = ap.polA.probs(), ap.polB.probs()

    rng = np.random.default_rng(7)
    unif = np.full((S, A), 1.0 / A)
    rand_A = rng.dirichlet(np.ones(A), size=S)
    rand_B = rng.dirichlet(np.ones(A), size=S)
    bz = boltzmann_advisor(panel, unif, BETA0, 1.0, TAU)[0]
    configs = [
        ("uniform + boltzmann @0.45", unif, bz, 0.45),
        ("random + random @0.30", rand_A, rand_B, 0.30),
        ("trained shaped @0.10", pi_A, pi_B, 0.10),
        ("trained shaped @0.45", pi_A, pi_B, 0.45),
    ]
    oracle_ok = run_oracle(panel, env, configs)

    print("=== UVFA Q-critic: ON-support all-action advantage error vs panel (ext stream) ===")
    print(f"    frozen shaped guide ({warm_iters} iters); fit on beta_grid={beta_grid}, "
          f"eval @ beta={eval_beta}; {n_episodes} eps, gae_lambda={GAE_LAMBDA}")
    print(f"    advisor support collapse: frac pi_B<1e-3 = "
          f"{float((pi_B < 1e-3).mean()):.3f}  (off-support is where the dial reads)\n")
    samp = collect_critic_samples(env, pi_A, pi_B, beta_grid, "ext",
                                  n_episodes=n_episodes, gae_lambda=GAE_LAMBDA,
                                  critic_iters=critic_iters, seed=seed)
    for m in m_list:
        mval = S if m == "tabular" else m
        crit = UVFAQCritic(S, A, mval, beta_grid, seed=seed)
        crit.fit(*samp)
        rows, pi_phi, A_exact, A_hat = stratified_adv_error(panel, crit, pi_A, pi_B, eval_beta)
        corr, mag, scale = occ_weighted_fidelity(panel, crit, pi_A, pi_B, eval_beta)
        tag = "tabular(Psi=I)" if (m == "tabular") else f"m={m}"
        print(f"  capacity {tag:14s} | rho-wtd corr(A_hat,A_exact)={corr:+.3f}  "
              f"mag-recovered={mag:.2f}  (A scale={scale:.4f})")
        print(f"                       | pi^phi bin       count   mean|dA|    max|dA|")
        for lab, cnt, mean_e, max_e in rows:
            print(f"                       |   {lab:12s} {cnt:6d}   {mean_e:8.4f}   {max_e:8.4f}")
        G_exact, G_hat = sampled_beta_dial(env, panel, crit, pi_A, pi_B, eval_beta,
                                           n_episodes=n_episodes, seed=seed + 1)
        ratio = G_hat / G_exact if abs(G_exact) > 1e-12 else float("nan")
        print(f"                       | end-to-end beta-dial: G_exact={G_exact:+.5f}  "
              f"G_hat={G_hat:+.5f}  ratio={ratio:.3f}\n")

    print(f"[smoke] oracle {'PASS' if oracle_ok else 'FAIL'}; critic on-support fidelity above. "
          f"Coverage-vs-capacity read: compare the m=16 (sharing) vs tabular (no sharing) "
          f"off-support rows.")
    return oracle_ok


def _disagreement(pi_A, pi_B, terminal_mask):
    D = 0.5 * np.abs(pi_A - pi_B).sum(axis=1)
    return D * (~terminal_mask)


def sampled_control_gradient(A_E_hat, A_I_hat, d_hat, pi_A, pi_B, beta, gamma, tau,
                             *, terminal_mask=None):
    g = gamma
    if terminal_mask is not None:
        A_E_hat = A_E_hat.copy(); A_E_hat[terminal_mask] = 0.0
        A_I_hat = A_I_hat.copy(); A_I_hat[terminal_mask] = 0.0
    G_beta = float(d_hat @ ((pi_A - pi_B) * A_E_hat).sum(axis=1))
    dJ_dbeta = -G_beta / (1 - g)
    EI = (pi_B * A_I_hat).sum(1); EE = (pi_B * A_E_hat).sum(1)
    cov = (pi_B * A_I_hat * A_E_hat).sum(1) - EI * EE
    H_lambda = float(d_hat @ cov)
    dJ_dlambda = beta / (tau * (1 - g)) * H_lambda
    return dict(G_beta=G_beta, dJ_dbeta=dJ_dbeta, H_lambda=H_lambda, dJ_dlambda=dJ_dlambda)


def fit_dial_critics(env, pi_A, pi_B, beta_grid, m, *, n_episodes, gae_lambda, critic_iters, seed, streams=("ext", "int")):
    crits = {}
    for st in streams:
        samp = collect_critic_samples(env, pi_A, pi_B, list(beta_grid), st, n_episodes=n_episodes, gae_lambda=gae_lambda, critic_iters=critic_iters, seed=seed)
        c = UVFAQCritic(env.n_states, env.n_actions, m, list(beta_grid), seed=seed)
        c.fit(*samp)
        crits[st] = c
    return crits


def sampled_dial(env, crits, pi_A, pi_B, beta, *, tau, n_episodes, seed):
    pi_phi = _mix(pi_A, pi_B, beta)
    A_E = crits["ext"].advantage(beta, pi_phi)
    A_I = crits["int"].advantage(beta, pi_phi) if "int" in crits else np.zeros_like(A_E)
    roll = collect_rollouts(env, pi_phi, n_episodes, seed=seed)
    d_hat = empirical_occupancy(roll, env.n_states, env.gamma)
    return sampled_control_gradient(A_E, A_I, d_hat, pi_A, pi_B, beta, env.gamma, tau)


def _sampled_JE(env, mix, gamma, episode_seeds):
    mix = mix / mix.sum(1, keepdims=True)
    cdf = np.cumsum(mix, axis=1)
    tot = 0.0
    for es in episode_seeds:
        r = np.random.default_rng(int(es))
        s = env.reset(seed=int(es)); disc, Jk = 1.0, 0.0
        while True:
            a = int(np.searchsorted(cdf[s], r.random()))
            ns, re, _, term, trunc, _ = env.step(a)
            Jk += disc * re; disc *= gamma; s = ns
            if term or trunc:
                break
        tot += Jk
    return tot / len(episode_seeds)


def dJ_ladder(env, panel, dial_dJdb, pi_A, pi_B, beta, *, gamma, n_episodes, seed):
    Ds = _disagreement(pi_A, pi_B, env.terminal_mask)
    roll = collect_rollouts(env, _mix(pi_A, pi_B, beta), n_episodes, seed=seed + 11)
    D_e = float(empirical_occupancy(roll, env.n_states, gamma) @ Ds)
    step1 = (1 - gamma) / gamma * DBUDGET / max(D_e, 1e-6)
    rng = np.random.default_rng(seed + 5)
    eps_seeds = rng.integers(1 << 31, size=n_episodes)
    Jb_x = panel.J_ext(_mix(pi_A, pi_B, beta))
    Jb_s = _sampled_JE(env, _mix(pi_A, pi_B, beta), gamma, eps_seeds)
    out = []
    for scale in (0.5, 1.0, 2.0):
        b2 = min(1.0, beta + scale * step1)
        pred = dial_dJdb * (b2 - beta)
        real_x = panel.J_ext(_mix(pi_A, pi_B, b2)) - Jb_x
        real_s = _sampled_JE(env, _mix(pi_A, pi_B, b2), gamma, eps_seeds) - Jb_s
        out.append((scale, b2 - beta, pred, real_x, real_s))
    return out


def probe_snapshot(env, panel, pi_A, pi_B, beta, m, beta_grid, *, tau, online_eps, converge_eps, critic_iters, seed):
    gx = panel.control_gradient(pi_A, pi_B, beta, tau=tau)
    c_on = fit_dial_critics(env, pi_A, pi_B, beta_grid, m, n_episodes=online_eps,
                            gae_lambda=GAE_LAMBDA, critic_iters=critic_iters, seed=seed)
    corr_on, _, _ = occ_weighted_fidelity(panel, c_on["ext"], pi_A, pi_B, beta)
    d_on = sampled_dial(env, c_on, pi_A, pi_B, beta, tau=tau, n_episodes=online_eps, seed=seed + 1)
    c_cv = fit_dial_critics(env, pi_A, pi_B, beta_grid, m, n_episodes=converge_eps,
                            gae_lambda=GAE_LAMBDA, critic_iters=critic_iters, seed=seed + 2)
    corr_cv, _, _ = occ_weighted_fidelity(panel, c_cv["ext"], pi_A, pi_B, beta)
    d_cv = sampled_dial(env, c_cv, pi_A, pi_B, beta, tau=tau, n_episodes=converge_eps, seed=seed + 3)
    ladder = dJ_ladder(env, panel, d_cv["dJ_dbeta"], pi_A, pi_B, beta,
                       gamma=env.gamma, n_episodes=online_eps, seed=seed + 4)
    pi_phi = _mix(pi_A, pi_B, beta)
    return dict(beta=beta, dJdb_exact=gx.dJ_dbeta,
                dJdb_on=d_on["dJ_dbeta"], corr_on=corr_on,
                dJdb_cv=d_cv["dJ_dbeta"], corr_cv=corr_cv,
                gap=d_on["dJ_dbeta"] - d_cv["dJ_dbeta"], ladder=ladder,
                minpiphi=float(pi_phi.min()),
                offsupp=float((pi_phi < 1e-2).mean()))


def run_static_probe(advisor_mode="shaped", *, held_beta=BETA0, lam=1.0, m_list=(16, "tabular"), n_iters=40, snapshots=(15, 39), seed=0, train_eps=40, critic_iters=15, online_eps=40, converge_eps=120, probe_beta_grid=(0.30, 0.45, 0.60), verbose=True):
    env, panel, room = build()
    S, A, g = env.n_states, env.n_actions, env.gamma
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    polA = LinearSoftmaxPolicy(fm, A, init="zero", seed=seed + 1)
    polB = LinearSoftmaxPolicy(fm, A, init="zero", seed=seed + 2) if advisor_mode == "shaped" else None
    rng = np.random.default_rng(seed)
    V_E, V_I = np.zeros(S), np.zeros(S)
    snaps = {}
    for e in range(max(snapshots) + 1):
        pi_A = polA.probs()
        pi_B = polB.probs() if polB is not None else boltzmann_advisor(panel, pi_A, held_beta, lam, TAU)[0]
        mixture = _mix(pi_A, pi_B, held_beta)
        roll = collect_rollouts(env, mixture, train_eps, seed=int(rng.integers(1 << 31)))
        V_E = fit_value_tabular(roll, "ext", g, S, gae_lambda=GAE_LAMBDA, n_iters=critic_iters, V_init=V_E)
        aE, _ = compute_gae(roll, V_E, "ext", g, GAE_LAMBDA); A_E_k, _ = bucket_by_sa(roll, aE, S, A)
        if advisor_mode == "shaped":
            V_I = fit_value_tabular(roll, "int", g, S, gae_lambda=GAE_LAMBDA, n_iters=critic_iters, V_init=V_I)
            aI, _ = compute_gae(roll, V_I, "int", g, GAE_LAMBDA); A_I_k, _ = bucket_by_sa(roll, aI, S, A)
            A_B_k = A_E_k + lam * A_I_k
        else:
            A_B_k = A_E_k
        d_hat = empirical_occupancy(roll, S, g)
        with np.errstate(divide="ignore", invalid="ignore"):
            sequenced_component_update(rng, advisor_mode, polA, polB, pi_A, pi_B, held_beta,
                                       A_E_k, A_B_k, d_hat, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI)
        if e in snapshots:
            snaps[e] = (polA.probs().copy(),
                        polB.probs().copy() if polB is not None else pi_B.copy())

    if verbose:
        print(f"\n=== STATIC PROBE (gate): sampled-dial vs certified exact dial, held beta={held_beta} ===")
        print(f"    bridge's own SAMPLED co-training ({advisor_mode}); dJ/dbeta fidelity, three reads, ladder")
    records = {}
    for it, (pi_A, pi_B) in snaps.items():
        for m in m_list:
            mval = S if m == "tabular" else m
            r = probe_snapshot(env, panel, pi_A, pi_B, held_beta, mval, probe_beta_grid,
                               tau=TAU, online_eps=online_eps, converge_eps=converge_eps,
                               critic_iters=critic_iters, seed=seed)
            records[(it, m)] = r
            if verbose:
                tag = "tabular" if m == "tabular" else f"m={m}"
                lad1 = next(l for l in r["ladder"] if l[0] == 1.0)
                print(f"  iter {it:4d} {tag:8s} | dJ/db exact={r['dJdb_exact']:+.4f} "
                      f"cv={r['dJdb_cv']:+.4f} on={r['dJdb_on']:+.4f} | corr_cv={r['corr_cv']:+.2f} "
                      f"corr_on={r['corr_on']:+.2f} gap={r['gap']:+.4f} | ladder 1x pred={lad1[2]:+.4f} "
                      f"real_x={lad1[3]:+.4f} | minPiPhi={r['minpiphi']:.3f}")
    if verbose:
        best = max(r["corr_cv"] for r in records.values())
        print(f"  -> GATE READ: best converge-then-read corr(A_hat,A_exact) = {best:+.2f} "
              f"(linear-FA dial is trustworthy to act on only if this clears ~0.9; "
              f"tabular is the no-sharing ceiling, full run needed for the collapsed off-support regime)")
    return records



def _resolve_outdir(outdir):
    import os
    outdir = outdir or os.path.join(os.getcwd(), "runs")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def run_bridge_sweep(*, beta0_grid=BETA0_GRID, m=16, advisor_mode="shaped", n_iters=40, snapshots=(15, 39), seed=0, train_eps=40, critic_iters=15, online_eps=40, converge_eps=120, corr_bar=0.9, mag_floor=0.02, outdir=None, name=None, verbose=True):
    import time
    t0 = time.time()
    rows = []
    for b0 in beta0_grid:
        grid = sorted({round(max(0.02, b0 - 0.10), 4), round(b0, 4),
                       round(min(0.98, b0 + 0.10), 4)})
        recs = run_static_probe(advisor_mode=advisor_mode, held_beta=b0, m_list=(m,),
                                n_iters=n_iters, snapshots=snapshots, seed=seed,
                                train_eps=train_eps, critic_iters=critic_iters,
                                online_eps=online_eps, converge_eps=converge_eps,
                                probe_beta_grid=tuple(grid), verbose=False)
        b0_rows = []
        for (it, _m), r in sorted(recs.items()):
            b0_rows.append(dict(beta0=b0, band=_band(b0), stage=it, mag=abs(r["dJdb_exact"]),
                                corr_cv=r["corr_cv"],
                                sign_ok=bool(np.sign(r["dJdb_cv"]) == np.sign(r["dJdb_exact"])),
                                offsupp=r["offsupp"], minpiphi=r["minpiphi"]))

        developed = max((rr["mag"] for rr in b0_rows), default=0.0) >= mag_floor
        for rr in b0_rows:
            rr["regime"] = ("unresolved" if not developed
                            else "magnitude" if rr["mag"] >= mag_floor else "sign/cross")
        rows.extend(b0_rows)
        if verbose:
            print(f"  [sweep] beta_0={b0:.2f} done ({len(rows)//max(len(snapshots),1)}/"
                  f"{len(beta0_grid)} beta_0) | elapsed {(time.time()-t0)/60:.0f} min", flush=True)

    if verbose:
        print(f"\n=== BRIDGE beta_0 SWEEP: (beta_0, stage) fidelity surface @ m={m} "
              f"| gate {GATE_BAND}, stress {STRESS_BAND} (fixed in advance) ===")
        print(" beta_0 band   | stage | |dJ/db|  regime     corr_cv  sign  offsupp")
        for r in rows:
            print(f"  {r['beta0']:.2f}  {r['band']:6s} | {r['stage']:5d} | "
                  f"{r['mag']:.4f}  {r['regime']:9s}  {r['corr_cv']:+.2f}   "
                  f"{'T' if r['sign_ok'] else 'F':1s}     {r['offsupp']:.2f}")

        def cells(band, regime):
            return [r for r in rows if r["band"] == band and r["regime"] == regime]
        g_mag = cells("gate", "magnitude")
        g_sgn = cells("gate", "sign/cross")
        g_unr = cells("gate", "unresolved")
        mag_pass = all(r["corr_cv"] >= corr_bar for r in g_mag) if g_mag else None
        sgn_pass = all(r["sign_ok"] for r in g_sgn) if g_sgn else None

        def verdict(p, n):
            return ("PASS" if p is True else "FAIL" if p is False
                    else "n/a (no resolved in-band cells at this budget)") + f"  ({n} cells)"
        print(f"\n  GATE (band {GATE_BAND}, decided here):")
        print(f"    magnitude bar  -- corr_cv >= {corr_bar} where |dJ/db| >= {mag_floor}: "
              f"{verdict(mag_pass, len(g_mag))}")
        print(f"    sign bar       -- sampled sign == exact sign at a genuine crossing: "
              f"{verdict(sgn_pass, len(g_sgn))}")
        if g_unr:
            print(f"    unresolved     -- {len(g_unr)} in-band cells never developed a slope "
                  f">= {mag_floor} (budget too low to test either bar there)")
        s_rows = [r for r in rows if r["band"] == "stress"]
        if s_rows:
            sc = np.mean([r["corr_cv"] for r in s_rows])
            print(f"  STRESS (band {STRESS_BAND}, NOT counted): mean corr_cv={sc:+.2f} over "
                  f"{len(s_rows)} cells -- reported as degradation under severe collapse, not against transfer")
        print("  NB tripwire: any FAIL above is INSIDE the deployable band and counts fully; "
              "the stress label excuses 0.6-0.8 only.")
    cfg = dict(experiment="bridge_beta0_sweep", advisor_mode=advisor_mode, m=m,
               beta0_grid=list(beta0_grid), gate_band=list(GATE_BAND), stress_band=list(STRESS_BAND),
               n_iters=n_iters, snapshots=list(snapshots), seed=seed, train_eps=train_eps,
               online_eps=online_eps, converge_eps=converge_eps, critic_iters=critic_iters,
               corr_bar=corr_bar, mag_floor=mag_floor, gae_lambda=GAE_LAMBDA, d_feat=D_FEAT,
               tau=TAU, instrument="room-field (exp_control.build)")
    outdir = _resolve_outdir(outdir)
    save_run(outdir, name or "bridge_beta0_sweep", cfg, rows)
    if verbose:
        print(f"  saved: {name or 'bridge_beta0_sweep'} ({len(rows)} surface cells)")
    return rows


def run_critic_comparison(*, held_beta=0.45, lam=1.0, advisor_mode="shaped", n_iters=2500, snapshots=(250, 800, 1600, 2499), seed=0, train_eps=200, critic_iters=40, probe_eps=500, critic_specs=("rand16", "rbf64", "tabular", "mlp_geom", "mlp_onehot", "cnn_local"), mlp_epochs=200, crop=7, outdir=None, name=None, verbose=True):
    import time
    env, panel, room = build()
    S, A = env.n_states, env.n_actions
    term_mask = np.asarray(env.terminal_mask, dtype=bool)
    if not _TORCH:
        critic_specs = tuple(s for s in critic_specs if s != "cnn_local")
        if verbose:
            print("  [note] torch absent -> cnn_local skipped; mlp_* use the numpy fallback")
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    ap = SampledApparatus(panel, env, fm, A, advisor_mode=advisor_mode, beta=held_beta,
                          lam=lam, tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                          distractor_states=room, seed=seed, n_episodes=train_eps,
                          gae_lambda=GAE_LAMBDA, critic_iters=critic_iters)
    snaps = {}; last = 0; t0 = time.time()
    for cp in sorted(snapshots):
        if cp > last:
            ap.run(cp - last); last = cp
        pi_A = ap.polA.probs().copy()
        pi_B = (ap.polB.probs().copy() if ap.polB is not None
                else ap._advisor_probs(ap.polA.probs(), held_beta, lam).copy())
        snaps[cp] = (pi_A, pi_B)
        if verbose:
            print(f"  [cotrain] snapshot {cp} | elapsed {(time.time()-t0)/60:.0f} min", flush=True)

    grid = sorted({round(max(0.02, held_beta - 0.10), 4), round(held_beta, 4),
                   round(min(0.98, held_beta + 0.10), 4)})
    probe_log = []; arrays = {}
    if verbose:
        print(f"\n=== CRITIC-CLASS COMPARISON: sampled dial recovery @ held beta={held_beta} ===")
        print("  stage  critic       | dJ/db exact   sampled    sign  corr   minPiPhi")
    for cp, (pi_A, pi_B) in snaps.items():
        pi_phi = _mix(pi_A, pi_B, held_beta); sol = panel.solve(pi_phi)
        grx = panel.control_gradient(pi_A, pi_B, held_beta, tau=TAU)
        sE = collect_critic_samples(env, pi_A, pi_B, grid, "ext", n_episodes=probe_eps,
                                    gae_lambda=GAE_LAMBDA, critic_iters=critic_iters, seed=seed)
        sI = collect_critic_samples(env, pi_A, pi_B, grid, "int", n_episodes=probe_eps,
                                    gae_lambda=GAE_LAMBDA, critic_iters=critic_iters, seed=seed + 1)
        d_hat = empirical_occupancy(collect_rollouts(env, pi_phi, probe_eps, seed=seed + 2), S, env.gamma)
        arrays[f"piA_it{cp}"] = pi_A; arrays[f"piB_it{cp}"] = pi_B
        arrays[f"Aext_exact_it{cp}"] = sol.A_ext; arrays[f"Aint_exact_it{cp}"] = sol.A_int
        arrays[f"dhat_it{cp}"] = d_hat; arrays[f"d_exact_it{cp}"] = sol.d
        for spec in critic_specs:
            cE = build_critic(spec, env, grid, seed=seed, mlp_epochs=mlp_epochs);
            cE.fit(*sE)
            cI = build_critic(spec, env, grid, seed=seed + 1, mlp_epochs=mlp_epochs);
            cI.fit(*sI)
            dl = sampled_control_gradient(cE.advantage(held_beta, pi_phi),
                                          cI.advantage(held_beta, pi_phi),
                                          d_hat, pi_A, pi_B, held_beta, env.gamma, TAU,
                                          terminal_mask=term_mask)
            corr, mag, _ = occ_weighted_fidelity(panel, cE, pi_A, pi_B, held_beta)
            probe_log.append(dict(stage=cp, beta0=held_beta, critic=spec,
                                  dJdb_exact=grx.dJ_dbeta, dJdb_hat=dl["dJ_dbeta"],
                                  sign_ok=int(np.sign(dl["dJ_dbeta"]) == np.sign(grx.dJ_dbeta)),
                                  corr=corr, mag_recovered=mag, minpiphi=float(pi_phi.min()),
                                  offsupp=float((pi_phi < 1e-2).mean())))
            arrays[f"Aexthat_{spec}_it{cp}"] = cE.advantage(held_beta, pi_phi)
            if verbose:
                print(f"  {cp:5d}  {spec:11s} | {grx.dJ_dbeta:+.4f}     {dl['dJ_dbeta']:+8.4f}   "
                      f"{'T' if probe_log[-1]['sign_ok'] else 'F'}    {corr:+.2f}   {pi_phi.min():.4f}")

    if any(s == "cnn_local" for s in critic_specs):
        arrays["local_view_obs"] = build_local_view(env, crop)
    cfg = dict(experiment="bridge_critic_comparison", advisor_mode=advisor_mode,
               held_beta=held_beta, lam=lam, n_iters=n_iters, snapshots=list(snapshots),
               seed=seed, train_eps=train_eps, probe_eps=probe_eps, critic_iters=critic_iters,
               gae_lambda=GAE_LAMBDA, critic_specs=list(critic_specs), d_feat=D_FEAT,
               tau=TAU, gamma=env.gamma, j_opt=J_OPT, crop=crop,
               cnn_channels="wallN,wallE,wallS,wallW,agent_center,oob,goal_tile_identity",
               dial_occupancy="sampled_dhat (exact d saved alongside)",
               torch=(torch.__version__ if _TORCH else None), device=_torch_device(),
               instrument="room-field (exp_control.build)")
    outdir = _resolve_outdir(outdir)
    nm = name or f"bridge_critic_cmp_b{int(round(held_beta * 100)):02d}"
    save_run(outdir, nm + "_train", cfg, ap.log, arrays=arrays)
    save_run(outdir, nm + "_probe", cfg, probe_log)
    if verbose:
        print(f"\n  saved: {nm}_train (per-iter training log + {len(arrays)} arrays), "
              f"{nm}_probe (critic-class probes)")
    return ap.log, probe_log


if __name__ == "__main__":
    import sys
    smoke()
    for hb in ([0.45] if "fast" in sys.argv else [0.2, 0.3, 0.4, 0.5]):
        run_critic_comparison(held_beta=hb, n_iters=2500, snapshots=(250, 800, 1600, 2499),
                              train_eps=200, probe_eps=500, critic_iters=40, seed=0)