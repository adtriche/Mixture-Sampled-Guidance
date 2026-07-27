from __future__ import annotations
import numpy as np


class FeatureMap:
    def __init__(self, Phi: np.ndarray) -> None:
        self.Phi = np.asarray(Phi, dtype=np.float64)
        self.S, self.d = self.Phi.shape
        self._pinv = None

    @property
    def pinv(self) -> np.ndarray:
        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.Phi)
        return self._pinv

    @classmethod
    def random(cls, S: int, d: int, seed=None) -> "FeatureMap":
        rng = np.random.default_rng(seed)
        return cls(rng.standard_normal((S, d)) / np.sqrt(d))

    @classmethod
    def tabular(cls, S: int) -> "FeatureMap":
        return cls(np.eye(S))


class LinearSoftmaxPolicy:
    def __init__(self, feature_map: FeatureMap, n_actions: int, init: str = "zero", seed=None) -> None:
        self.fm = feature_map
        self.A = int(n_actions)
        if init == "zero":
            self.W = np.zeros((self.A, feature_map.d), dtype=np.float64)
        elif init == "random":
            rng = np.random.default_rng(seed)
            self.W = 0.01 * rng.standard_normal((self.A, feature_map.d))
        else:
            raise ValueError("init must be 'zero' or 'random'")

    def logits(self) -> np.ndarray:
        return self.fm.Phi @ self.W.T

    def probs(self) -> np.ndarray:
        z = self.logits()
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def grad_expected_coeff(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=np.float64)
        pi = self.probs()
        cbar = (pi * c).sum(axis=1, keepdims=True)
        M = pi * (c - cbar)                       # (S, A)
        return M.T @ self.fm.Phi                  # (A, d)

    def apply_grad(self, g: np.ndarray, lr: float) -> None:
        self.W = self.W + lr * np.asarray(g, dtype=np.float64)

    def fit_logits(self, target_logits: np.ndarray) -> float:
        target_logits = np.asarray(target_logits, dtype=np.float64)
        self.W = (self.fm.pinv @ target_logits).T
        return float(np.max(np.abs(self.fm.Phi @ self.W.T - target_logits)))

    def fit_probs(self, target_probs: np.ndarray) -> float:
        target_probs = np.asarray(target_probs, dtype=np.float64)
        logits = np.log(np.clip(target_probs, 1e-300, None))
        logits = logits - logits.mean(axis=1, keepdims=True)
        return self.fit_logits(logits)


def capped_simplex_project(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, iters: int = 60) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    nu_lo = (y - hi).min(axis=1) - 1.0
    nu_hi = (y - lo).max(axis=1) + 1.0
    for _ in range(iters):
        nu = 0.5 * (nu_lo + nu_hi)
        s = np.clip(y - nu[:, None], lo, hi).sum(axis=1)
        too_big = s > 1.0
        nu_lo = np.where(too_big, nu, nu_lo)
        nu_hi = np.where(~too_big, nu, nu_hi)
    nu = 0.5 * (nu_lo + nu_hi)
    return np.clip(y - nu[:, None], lo, hi)


def _component_update(policy, pi_other0, self_weight, A_stream_k, d_phi, *, lr, delta_c, delta_phi, prior_phi_disp=None):
    pi_self0 = policy.probs()
    pi_other0 = np.asarray(pi_other0)
    pi_phi0 = self_weight * pi_self0 + (1.0 - self_weight) * pi_other0
    Delta = np.zeros_like(pi_self0) if prior_phi_disp is None else np.asarray(prior_phi_disp)

    g = self_weight * policy.grad_expected_coeff(d_phi[:, None] * A_stream_k)

    W_save = policy.W.copy()
    policy.W = W_save + lr * g
    pi_cand = policy.probs()
    policy.W = W_save

    comp_lo = pi_self0 * (1.0 - delta_c)
    comp_hi = pi_self0 * (1.0 + delta_c)
    mix_lo = pi_self0 + pi_phi0 * (-delta_phi - Delta) / self_weight
    mix_hi = pi_self0 + pi_phi0 * (delta_phi - Delta) / self_weight
    lo = np.clip(np.maximum(comp_lo, mix_lo), 0.0, None)
    hi = np.minimum(comp_hi, mix_hi)
    pi_star = capped_simplex_project(pi_cand, lo, hi)

    policy.fit_probs(pi_star)
    pi_new = policy.probs()

    r_self = pi_new / pi_self0
    pi_phi_cur = pi_phi0 * (1.0 + Delta) + self_weight * (pi_new - pi_self0)
    r_phi = pi_phi_cur / pi_phi0
    return {
        "max_r_dev": float(np.max(np.abs(r_self - 1.0))),
        "max_rphi_dev": float(np.max(np.abs(r_phi - 1.0))),
        "phi_disp": r_phi - 1.0,
    }


def actor_update(policy_A, pi_B, beta, A_E_k, d_phi, *, lr, delta_c, delta_phi, prior_phi_disp=None):
    out = _component_update(policy_A, pi_B, 1.0 - beta, A_E_k, d_phi, lr=lr, delta_c=delta_c, delta_phi=delta_phi, prior_phi_disp=prior_phi_disp)
    return {"max_rA_dev": out["max_r_dev"], "max_rphi_dev": out["max_rphi_dev"], "phi_disp": out["phi_disp"]}


def advisor_update(policy_B, pi_A, beta, A_B_k, d_phi, *, lr, delta_c, delta_phi, prior_phi_disp=None):
    out = _component_update(policy_B, pi_A, beta, A_B_k, d_phi, lr=lr, delta_c=delta_c, delta_phi=delta_phi, prior_phi_disp=prior_phi_disp)
    return {"max_rB_dev": out["max_r_dev"], "max_rphi_dev": out["max_rphi_dev"], "phi_disp": out["phi_disp"]}


def boltzmann_advisor(panel, pi_A, beta, lam, tau, *, iters=300, tol=1e-12, damping=0.0):
    pi_A = np.asarray(pi_A)
    S, A = pi_A.shape
    pi_B = np.full((S, A), 1.0 / A)
    res = np.inf
    for _ in range(iters):
        sol = panel.solve((1.0 - beta) * pi_A + beta * pi_B)
        z = (sol.Q_ext + lam * sol.Q_int) / tau
        z -= z.max(axis=1, keepdims=True)
        pi_B_new = np.exp(z)
        pi_B_new /= pi_B_new.sum(axis=1, keepdims=True)
        if damping > 0.0:
            pi_B_new = (1.0 - damping) * pi_B_new + damping * pi_B
        res = float(np.max(np.abs(pi_B_new - pi_B)))
        pi_B = pi_B_new
        if res < tol:
            break
    return pi_B, res


def rb_update(policy_RB, lam, A_E_RB, A_I_RB, d_RB, *, lr, delta_c=None):
    pi0 = policy_RB.probs()
    g = policy_RB.grad_expected_coeff(d_RB[:, None] * (A_E_RB + lam * A_I_RB))
    if delta_c is None:
        policy_RB.apply_grad(g, lr)
    else:
        W_save = policy_RB.W.copy()
        policy_RB.W = W_save + lr * g
        pi_cand = policy_RB.probs()
        policy_RB.W = W_save
        lo = np.clip(pi0 * (1.0 - delta_c), 0.0, None)
        hi = pi0 * (1.0 + delta_c)
        policy_RB.fit_probs(capped_simplex_project(pi_cand, lo, hi))
    pi_new = policy_RB.probs()
    return {"max_rRB_dev": float(np.max(np.abs(pi_new / pi0 - 1.0)))}

def sequenced_component_update(rng, mode, polA, polB, pi_A_roll, pi_B_roll, beta, A_E_k, A_B_k, d_phi, *, lr, delta_c, delta_phi):
    if mode in ("none", "boltzmann"):
        ai = actor_update(polA, pi_B_roll, beta, A_E_k, d_phi,
                          lr=lr, delta_c=delta_c, delta_phi=delta_phi)
        return {"max_rA_dev": ai["max_rA_dev"], "max_rphi_dev": ai["max_rphi_dev"]}

    actor_first = bool(rng.integers(2))
    if actor_first:
        ai = actor_update(polA, pi_B_roll, beta, A_E_k, d_phi,
                          lr=lr, delta_c=delta_c, delta_phi=delta_phi)
        bi = advisor_update(polB, pi_A_roll, beta, A_B_k, d_phi,
                            lr=lr, delta_c=delta_c, delta_phi=delta_phi,
                            prior_phi_disp=ai["phi_disp"])
        joint = bi["max_rphi_dev"]
    else:
        bi = advisor_update(polB, pi_A_roll, beta, A_B_k, d_phi,
                            lr=lr, delta_c=delta_c, delta_phi=delta_phi)
        ai = actor_update(polA, pi_B_roll, beta, A_E_k, d_phi,
                          lr=lr, delta_c=delta_c, delta_phi=delta_phi,
                          prior_phi_disp=bi["phi_disp"])
        joint = ai["max_rphi_dev"]
    return {"max_rA_dev": ai["max_rA_dev"], "max_rB_dev": bi["max_rB_dev"],
            "max_rphi_dev": joint, "order": "actor_first" if actor_first else "advisor_first"}


class Apparatus:
    def __init__(self, panel, feature_map, n_actions, *, advisor_mode="disjoint",
                 beta=0.4, lam=1.0, tau=0.7, lr=1.0, delta_c=0.1, delta_phi=0.05,
                 with_rb=False, distractor_states=None, seed=0,
                 actor_init="zero", advisor_init="zero", rb_init="zero"):
        if advisor_mode not in ("none", "disjoint", "shaped", "boltzmann"):
            raise ValueError(f"unknown advisor_mode {advisor_mode!r}")
        self.panel = panel
        self.fm = feature_map
        self.A = int(n_actions)
        self.mode = advisor_mode
        self.beta, self.lam, self.tau = float(beta), float(lam), float(tau)
        self.lr, self.delta_c, self.delta_phi = float(lr), float(delta_c), float(delta_phi)
        self.with_rb = bool(with_rb)
        self.distractor = None if distractor_states is None else np.asarray(distractor_states)
        self.rng = np.random.default_rng(seed)
        self.polA = LinearSoftmaxPolicy(feature_map, n_actions, init=actor_init, seed=seed + 1)
        self.polB = (None if advisor_mode in ("none", "boltzmann")
                     else LinearSoftmaxPolicy(feature_map, n_actions, init=advisor_init, seed=seed + 2))
        self.polR = (LinearSoftmaxPolicy(feature_map, n_actions, init=rb_init, seed=seed + 3)
                     if with_rb else None)
        self.log = []
        self.t = 0

    def _advisor_probs(self, pi_A, beta, lam):
        if self.mode == "none":
            return pi_A
        if self.mode == "boltzmann":
            return boltzmann_advisor(self.panel, pi_A, beta, lam, self.tau)[0]
        return self.polB.probs()

    def _assigned_advantage(self, A_E_k, A_I_k, lam):
        return A_I_k if self.mode == "disjoint" else (A_E_k + lam * A_I_k)

    def _read_panel(self, beta, lam, pi_A, pi_B, sol_phi):
        rec = {"iter": self.t, "beta": beta, "lam": lam,
               "J_ext_actor": self.panel.J_ext(pi_A),
               "J_ext_mix": sol_phi.J_ext,
               "J_ext_advisor": self.panel.J_ext(pi_B)}
        if self.with_rb:
            rec["J_ext_rb"] = self.panel.J_ext(self.polR.probs())
        if self.distractor is not None:
            rec["distr_occ_actor"] = float(self.panel.solve(pi_A).d[self.distractor].sum())
            rec["distr_occ_mix"] = float(sol_phi.d[self.distractor].sum())
            rec["distr_occ_advisor"] = float(self.panel.solve(pi_B).d[self.distractor].sum())
            if self.with_rb:
                rec["distr_occ_rb"] = float(self.panel.solve(self.polR.probs()).d[self.distractor].sum())
        m = self.panel.meter(pi_A, pi_B, beta)
        rec["D_e"], rec["I_e"] = m.D_e, m.I_e
        tau = self.tau if self.mode in ("shaped", "boltzmann") else None
        gr = self.panel.control_gradient(pi_A, pi_B, beta, tau=tau)
        rec["G_beta"], rec["dJ_dbeta"] = gr.G_beta, gr.dJ_dbeta
        rec["H_lambda"], rec["dJ_dlambda"] = gr.H_lambda, gr.dJ_dlambda
        return rec

    def step(self, beta=None, lam=None):
        beta = self.beta if beta is None else float(beta)
        lam = self.lam if lam is None else float(lam)
        if self.mode == "none":
            beta = 0.0

        pi_A = self.polA.probs()
        pi_B = self._advisor_probs(pi_A, beta, lam)
        sol_phi = self.panel.solve((1.0 - beta) * pi_A + beta * pi_B)
        A_E_k, A_I_k, d_phi = sol_phi.A_ext, sol_phi.A_int, sol_phi.d

        rec = self._read_panel(beta, lam, pi_A, pi_B, sol_phi)

        if self.with_rb:
            sR = self.panel.solve(self.polR.probs())
            rb_update(self.polR, lam, sR.A_ext, sR.A_int, sR.d, lr=self.lr, delta_c=self.delta_c)

        A_B_k = self._assigned_advantage(A_E_k, A_I_k, lam)
        rec.update(sequenced_component_update(
            self.rng, self.mode, self.polA, self.polB, pi_A, pi_B, beta,
            A_E_k, A_B_k, d_phi, lr=self.lr, delta_c=self.delta_c, delta_phi=self.delta_phi))

        self.log.append(rec)
        self.t += 1
        return rec

    def run(self, n_iters, beta_schedule=None, lam_schedule=None):
        for e in range(n_iters):
            b = None if beta_schedule is None else float(beta_schedule[e])
            l = None if lam_schedule is None else float(lam_schedule[e])
            self.step(b, l)
        return self.log