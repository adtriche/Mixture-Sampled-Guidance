import numpy as np
from apparatus import (LinearSoftmaxPolicy, boltzmann_advisor, rb_update, sequenced_component_update)
from rollout import (collect_rollouts, empirical_occupancy, compute_gae, bucket_by_sa, fit_value_tabular)


def discounted_return(roll, stream, gamma):
    r = roll.reward(stream)
    tot = 0.0
    for ep in range(roll.n_episodes):
        idx = np.flatnonzero(roll.ep_id == ep)
        tot += float(np.dot(gamma ** np.arange(idx.size), r[idx]))
    return tot / roll.n_episodes


class SampledApparatus:
    def __init__(self, panel, env, feature_map, n_actions, *, advisor_mode="disjoint",
                 beta=0.4, lam=1.0, tau=0.7, lr=1.0, delta_c=0.1, delta_phi=0.05,
                 with_rb=False, distractor_states=None, seed=0,
                 n_episodes=400, gae_lambda=0.95, critic_iters=40,
                 actor_init="zero", advisor_init="zero", rb_init="zero"):
        if advisor_mode not in ("none", "disjoint", "shaped", "boltzmann"):
            raise ValueError(f"unknown advisor_mode {advisor_mode!r}")
        self.panel, self.env, self.fm = panel, env, feature_map
        self.A, self.S, self.gamma = int(n_actions), env.n_states, env.gamma
        self.mode = advisor_mode
        self.beta, self.lam, self.tau = float(beta), float(lam), float(tau)
        self.lr, self.delta_c, self.delta_phi = float(lr), float(delta_c), float(delta_phi)
        self.with_rb = bool(with_rb)
        self.distractor = None if distractor_states is None else np.asarray(distractor_states)
        self.n_episodes = int(n_episodes)
        self.gae_lambda, self.critic_iters = float(gae_lambda), int(critic_iters)
        self.rng = np.random.default_rng(seed)
        self.polA = LinearSoftmaxPolicy(feature_map, n_actions, init=actor_init, seed=seed + 1)
        self.polB = (None if advisor_mode in ("none", "boltzmann")
                     else LinearSoftmaxPolicy(feature_map, n_actions, init=advisor_init, seed=seed + 2))
        self.polR = (LinearSoftmaxPolicy(feature_map, n_actions, init=rb_init, seed=seed + 3)
                     if with_rb else None)
        self.V_E = np.zeros(self.S); self.V_I = np.zeros(self.S)
        self.V_E_RB = np.zeros(self.S); self.V_I_RB = np.zeros(self.S)
        self.log = []; self.t = 0

    def _advisor_probs(self, pi_A, beta, lam):
        if self.mode == "none":
            return pi_A
        if self.mode == "boltzmann":
            return boltzmann_advisor(self.panel, pi_A, beta, lam, self.tau)[0]
        return self.polB.probs()

    def _assigned_advantage(self, A_E_k, A_I_k, lam):
        return A_I_k if self.mode == "disjoint" else (A_E_k + lam * A_I_k)

    def _seed(self):
        return int(self.rng.integers(1 << 31))

    def _fit_and_advantage(self, roll, stream, V):
        V = fit_value_tabular(roll, stream, self.gamma, self.S,
                              gae_lambda=self.gae_lambda, n_iters=self.critic_iters, V_init=V)
        adv, _ = compute_gae(roll, V, stream, self.gamma, self.gae_lambda)
        A_k, _ = bucket_by_sa(roll, adv, self.S, self.A)
        return V, A_k

    def _diagnostics(self, beta, lam, pi_A, pi_B, mixture, roll):
        rec = {"iter": self.t, "beta": beta, "lam": lam,
               "J_ext_actor": self.panel.J_ext(pi_A),
               "J_ext_mix": self.panel.J_ext(mixture),
               "J_ext_advisor": self.panel.J_ext(pi_B),
               "J_ext_mix_hat": discounted_return(roll, "ext", self.gamma)}
        if self.with_rb:
            rec["J_ext_rb"] = self.panel.J_ext(self.polR.probs())
        if self.distractor is not None:
            rec["distr_occ_actor"] = float(self.panel.solve(pi_A).d[self.distractor].sum())
            rec["distr_occ_mix"] = float(self.panel.solve(mixture).d[self.distractor].sum())
            rec["distr_occ_advisor"] = float(self.panel.solve(pi_B).d[self.distractor].sum())
            rec["distr_occ_mix_hat"] = float(
                empirical_occupancy(roll, self.S, self.gamma)[self.distractor].sum())
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
        mixture = (1.0 - beta) * pi_A + beta * pi_B

        roll = collect_rollouts(self.env, mixture, self.n_episodes, seed=self._seed())
        self.V_E, A_E_k = self._fit_and_advantage(roll, "ext", self.V_E)
        need_intrinsic = self.mode in ("disjoint", "shaped")
        if need_intrinsic:
            self.V_I, A_I_k = self._fit_and_advantage(roll, "int", self.V_I)
        else:
            A_I_k = np.zeros((self.S, self.A))
        d_phi = empirical_occupancy(roll, self.S, self.gamma)

        rec = self._diagnostics(beta, lam, pi_A, pi_B, mixture, roll)

        if self.with_rb:
            rollR = collect_rollouts(self.env, self.polR.probs(), self.n_episodes, seed=self._seed())
            self.V_E_RB, A_E_R = self._fit_and_advantage(rollR, "ext", self.V_E_RB)
            self.V_I_RB, A_I_R = self._fit_and_advantage(rollR, "int", self.V_I_RB)
            d_R = empirical_occupancy(rollR, self.S, self.gamma)
            rb_update(self.polR, lam, A_E_R, A_I_R, d_R, lr=self.lr, delta_c=self.delta_c)

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