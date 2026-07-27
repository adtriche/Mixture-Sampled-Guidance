from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
@dataclass
class PolicySolve:
    pi: np.ndarray
    P_pi: np.ndarray
    d: np.ndarray
    rho: np.ndarray
    V_ext: np.ndarray
    V_int: np.ndarray
    Q_ext: np.ndarray
    Q_int: np.ndarray
    A_ext: np.ndarray
    A_int: np.ndarray
    J_ext: float
    J_int: float

@dataclass
class MeterResult:
    beta: float
    D_s: np.ndarray
    D_e: float
    I_e: float
    tv_state: float
    tv_state_bound: float
    tv_sa: float
    tv_sa_bound: float

@dataclass
class GradientResult:
    beta: float
    G_beta: float
    H_lambda: float
    dJ_dbeta: float
    dJ_dlambda: Optional[float]
    delta_wd: float
    A_max: float
    G_beta_bound: float
    H_lambda_cs_bound: float
    grad: Tuple[float, Optional[float]]

@dataclass
class RatioResult:
    beta: float
    w_A: np.ndarray
    w_B: np.ndarray
    r_A: np.ndarray
    r_phi: np.ndarray
    r_Aphi: np.ndarray
    A_max: float
    mt_mean: float
    mt_second_moment: float
    mt_var: float
    ct_mean: float
    ct_second_moment: float
    ct_var: float
    mt_uniform_sm_bound: Optional[float]


@dataclass
class RecoveryResult:
    beta: float
    L_phi: float
    L_bar: float
    L_A: float
    occ_channel: float
    occ_bound: Optional[float]
    int_channel: float
    int_bound: Optional[float]
    total: float
    total_bound: Optional[float]
    I_e: float
    A_max: float


class Panel:
    def __init__(self, P, R_ext, R_int, gamma, mu0, terminal_mask=None):
        self.P = np.asarray(P, dtype=np.float64)
        self.R_ext = np.asarray(R_ext, dtype=np.float64)
        self.R_int = np.asarray(R_int, dtype=np.float64)
        self.gamma = float(gamma)
        self.mu0 = np.asarray(mu0, dtype=np.float64)
        self.S, self.A = self.R_ext.shape
        self.terminal_mask = (
            np.zeros(self.S, dtype=bool)
            if terminal_mask is None
            else np.asarray(terminal_mask, dtype=bool)
        )
        self._I = np.eye(self.S)

    @classmethod
    def from_env(cls, env):
        return cls(env.P, env.R_ext, env.R_int, env.gamma, env.mu0, env.terminal_mask)

    def induced_kernel(self, pi: np.ndarray) -> np.ndarray:
        return np.einsum("sa,sax->sx", pi, self.P)

    def _value(self, P_pi: np.ndarray, R_pi: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self._I - self.gamma * P_pi, R_pi)

    def _occupancy(self, P_pi: np.ndarray) -> np.ndarray:
        return np.linalg.solve(
            (self._I - self.gamma * P_pi).T, (1.0 - self.gamma) * self.mu0
        )

    def _q_from_v(self, R: np.ndarray, V: np.ndarray) -> np.ndarray:
        return R + self.gamma * np.einsum("sax,x->sa", self.P, V)

    def solve(self, pi: np.ndarray) -> PolicySolve:
        pi = np.asarray(pi, dtype=np.float64)
        P_pi = self.induced_kernel(pi)
        d = self._occupancy(P_pi)
        rho = d[:, None] * pi
        R_ext_pi = np.einsum("sa,sa->s", pi, self.R_ext)
        R_int_pi = np.einsum("sa,sa->s", pi, self.R_int)
        V_ext = self._value(P_pi, R_ext_pi)
        V_int = self._value(P_pi, R_int_pi)
        Q_ext = self._q_from_v(self.R_ext, V_ext)
        Q_int = self._q_from_v(self.R_int, V_int)
        A_ext = Q_ext - V_ext[:, None]
        A_int = Q_int - V_int[:, None]
        J_ext = float(self.mu0 @ V_ext)
        J_int = float(self.mu0 @ V_int)
        return PolicySolve(
            pi, P_pi, d, rho, V_ext, V_int, Q_ext, Q_int, A_ext, A_int, J_ext, J_int
        )

    def statewise_disagreement(
        self, pi_A: np.ndarray, pi_B: np.ndarray, respect_terminal: bool = True
    ) -> np.ndarray:
        D = 0.5 * np.abs(np.asarray(pi_A) - np.asarray(pi_B)).sum(axis=1)
        if respect_terminal:
            D = D * (~self.terminal_mask)
        return D

    def meter(
        self, pi_A: np.ndarray, pi_B: np.ndarray, beta: float, *, respect_terminal: bool = True
    ) -> MeterResult:
        pi_A = np.asarray(pi_A, dtype=np.float64)
        pi_B = np.asarray(pi_B, dtype=np.float64)
        sol_phi = self.solve(self.mixture(pi_A, pi_B, beta))
        sol_A = self.solve(pi_A)
        D_s = self.statewise_disagreement(pi_A, pi_B, respect_terminal)
        D_e = float(sol_phi.d @ D_s)
        I_e = beta * D_e
        g = self.gamma
        tv_state = 0.5 * np.abs(sol_phi.d - sol_A.d).sum()
        tv_sa = 0.5 * np.abs(sol_phi.rho - sol_A.rho).sum()
        return MeterResult(
            beta, D_s, D_e, I_e,
            tv_state, g / (1 - g) * I_e,
            tv_sa, 1 / (1 - g) * I_e,
        )

    def control_gradient(
        self, pi_A: np.ndarray, pi_B: np.ndarray, beta: float, *, tau: Optional[float] = None
    ) -> GradientResult:
        pi_A = np.asarray(pi_A, dtype=np.float64)
        pi_B = np.asarray(pi_B, dtype=np.float64)
        sol = self.solve(self.mixture(pi_A, pi_B, beta))
        AE, AI, d = sol.A_ext, sol.A_int, sol.d
        g = self.gamma

        g_state = ((pi_A - pi_B) * AE).sum(axis=1)
        G_beta = float(d @ g_state)
        dJ_dbeta = -G_beta / (1 - g)
        delta_wd = beta / (1 - g) * G_beta
        A_max = float(np.max(np.abs(AE)))
        D_beta = float(d @ self.statewise_disagreement(pi_A, pi_B))
        G_beta_bound = 2 * A_max * D_beta

        EI = (pi_B * AI).sum(axis=1)
        EE = (pi_B * AE).sum(axis=1)
        cov = (pi_B * AI * AE).sum(axis=1) - EI * EE
        H_lambda = float(d @ cov)
        varI = np.clip((pi_B * AI * AI).sum(axis=1) - EI * EI, 0.0, None)
        varE = np.clip((pi_B * AE * AE).sum(axis=1) - EE * EE, 0.0, None)
        H_cs = float(d @ (np.sqrt(varI) * np.sqrt(varE)))

        dJ_dlambda = (beta / (tau * (1 - g)) * H_lambda) if tau is not None else None
        grad = (dJ_dbeta, dJ_dlambda)
        return GradientResult(
            beta, G_beta, H_lambda, dJ_dbeta, dJ_dlambda, delta_wd,
            A_max, G_beta_bound, H_cs, grad,
        )

    def _ratios(self, pi_A, pi_B, beta, pi_A_plus, pi_B_plus):
        pi_phi = self.mixture(pi_A, pi_B, beta)
        pi_phi_plus = self.mixture(pi_A_plus, pi_B_plus, beta)
        r_A = pi_A_plus / pi_A
        r_B = pi_B_plus / pi_B
        r_phi = pi_phi_plus / pi_phi
        r_Aphi = pi_A_plus / pi_phi
        w_A = (1.0 - beta) * pi_A / pi_phi
        w_B = beta * pi_B / pi_phi
        return pi_phi, pi_phi_plus, r_A, r_B, r_phi, r_Aphi, w_A, w_B

    def ratio_diagnostics(
        self, pi_A, pi_B, beta, pi_A_plus, *, delta_phi: Optional[float] = None
    ) -> RatioResult:
        pi_A = np.asarray(pi_A, dtype=np.float64)
        pi_B = np.asarray(pi_B, dtype=np.float64)
        pi_A_plus = np.asarray(pi_A_plus, dtype=np.float64)
        sol = self.solve(self.mixture(pi_A, pi_B, beta))
        AE, rho = sol.A_ext, sol.rho
        _, _, r_A, _, r_phi, r_Aphi, w_A, w_B = self._ratios(
            pi_A, pi_B, beta, pi_A_plus, pi_B
        )
        A_max = float(np.max(np.abs(AE)))

        def moments(r):
            X = r * AE
            mean = float((rho * X).sum())
            sm = float((rho * X * X).sum())
            return mean, sm, sm - mean * mean

        mt_mean, mt_sm, mt_var = moments(r_phi)
        ct_mean, ct_sm, ct_var = moments(r_Aphi)
        mt_bound = (1.0 + delta_phi) ** 2 * A_max ** 2 if delta_phi is not None else None
        return RatioResult(
            beta, w_A, w_B, r_A, r_phi, r_Aphi, A_max,
            mt_mean, mt_sm, mt_var, ct_mean, ct_sm, ct_var, mt_bound,
        )

    def surrogate_recovery(
        self, pi_A, pi_B, beta, pi_A_plus, pi_B_plus=None, *,
        delta_c: Optional[float] = None, delta_phi: Optional[float] = None,
    ) -> RecoveryResult:
        pi_A = np.asarray(pi_A, dtype=np.float64)
        pi_B = np.asarray(pi_B, dtype=np.float64)
        pi_A_plus = np.asarray(pi_A_plus, dtype=np.float64)
        pi_B_plus = pi_B if pi_B_plus is None else np.asarray(pi_B_plus, dtype=np.float64)

        sol_phi = self.solve(self.mixture(pi_A, pi_B, beta))
        sol_A = self.solve(pi_A)
        AE, dphi, dA = sol_phi.A_ext, sol_phi.d, sol_A.d
        _, pi_phi_plus, r_A, r_B, r_phi, r_Aphi, w_A, w_B = self._ratios(
            pi_A, pi_B, beta, pi_A_plus, pi_B_plus
        )

        L_phi = float((dphi[:, None] * pi_phi_plus * AE).sum())
        L_bar = float((dA[:, None] * pi_A * r_phi * AE).sum())
        L_A = float((dA[:, None] * pi_A_plus * AE).sum())
        occ = L_phi - L_bar
        intg = L_bar - L_A
        total = L_phi - L_A

        I_e = beta * float(dphi @ self.statewise_disagreement(pi_A, pi_B))
        A_max = float(np.max(np.abs(AE)))
        g = self.gamma
        occ_bound = (
            2.0 * (1.0 + delta_phi) * A_max / (1.0 - g) * I_e
            if delta_phi is not None else None
        )
        int_bound = (
            2.0 * beta / (1.0 - beta) * delta_c * A_max
            if (delta_c is not None and beta < 1.0) else None
        )
        total_bound = (
            occ_bound + int_bound
            if (occ_bound is not None and int_bound is not None) else None
        )
        return RecoveryResult(
            beta, L_phi, L_bar, L_A, occ, occ_bound, intg, int_bound,
            total, total_bound, I_e, A_max,
        )

    @staticmethod
    def schedule_budgets(I_e_seq, beta_seq):
        I_e_seq = np.asarray(I_e_seq, dtype=np.float64)
        beta_seq = np.asarray(beta_seq, dtype=np.float64)
        return float(I_e_seq.sum()), float((beta_seq / (1.0 - beta_seq)).sum())

    @staticmethod
    def mixture(pi_A: np.ndarray, pi_B: np.ndarray, beta: float) -> np.ndarray:
        return (1.0 - beta) * np.asarray(pi_A, dtype=np.float64) + beta * np.asarray(
            pi_B, dtype=np.float64
        )

    def J_ext(self, pi: np.ndarray) -> float:
        return self.solve(pi).J_ext

    def J_int(self, pi: np.ndarray) -> float:
        return self.solve(pi).J_int