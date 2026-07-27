import numpy as np
from gridworld_env import GridEnv
from panel import Panel
from rollout import collect_rollouts, empirical_occupancy, compute_gae, bucket_by_sa
from recording import save_run

SIZE, START, GOAL, DCELL, R_INT = 10, (0, 0), (9, 9), (9, 0), 0.04
TAU = 0.30
DELTA_C = 0.10
DELTA_PHI = 0.05
BETA_GRID = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
SEED = 0
N_CONFIGS = 400
GAE_LAMBDA = 0.95
SAMPLED_BUDGETS = [1000, 5000, 20000]
WITNESS_DC = 0.08
WITNESS_BETAS = [0.2, 0.5, 0.8]
OUTDIR = "runs/calib"

def random_policy(S, A, rng, temp=1.0):
    z = rng.standard_normal((S, A)) / temp
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def boltzmann(Q, tau):
    z = Q / tau
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def near_greedy(Q, eps):
    S, A = Q.shape
    pi = np.full((S, A), eps / A)
    pi[np.arange(S), Q.argmax(1)] += 1.0 - eps
    return pi


def vi_Q(P, R, gamma, iters=8000, tol=1e-11):
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(iters):
        Q = R + gamma * np.einsum("sax,x->sa", P, V)
        Vn = Q.max(1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return R + gamma * np.einsum("sax,x->sa", P, V)


def single_state_panel(gamma, A_E):
    A = len(A_E)
    P = np.ones((1, A, 1))
    R_ext = np.array([A_E], dtype=float)
    R_int = np.zeros((1, A))
    mu0 = np.array([1.0])
    return Panel(P, R_ext, R_int, gamma, mu0)


def check_bounds(panel, actor_opt, advisor_seek, rng):
    S, A = actor_opt.shape
    rows, worst_ratio = [], 0.0
    for _ in range(N_CONFIGS):
        pA = random_policy(S, A, rng, temp=float(rng.uniform(0.4, 2.5)))
        pB = random_policy(S, A, rng, temp=float(rng.uniform(0.4, 2.5)))
        beta = float(rng.uniform(0.05, 0.95))
        m = panel.meter(pA, pB, beta)
        rs = m.tv_state / m.tv_state_bound if m.tv_state_bound > 0 else 0.0
        rsa = m.tv_sa / m.tv_sa_bound if m.tv_sa_bound > 0 else 0.0
        worst_ratio = max(worst_ratio, rs, rsa)
        rows.append(dict(regime="random", beta=beta, D_e=m.D_e, I_e=m.I_e,
                         tv_state=m.tv_state, tv_state_bound=m.tv_state_bound,
                         tv_sa=m.tv_sa, tv_sa_bound=m.tv_sa_bound,
                         slack_state=m.tv_state_bound - m.tv_state,
                         slack_sa=m.tv_sa_bound - m.tv_sa,
                         violated=int(m.tv_state > m.tv_state_bound + 1e-9
                                      or m.tv_sa > m.tv_sa_bound + 1e-9)))
    for beta in BETA_GRID:
        m = panel.meter(actor_opt, advisor_seek, beta)
        rows.append(dict(regime="operating", beta=beta, D_e=m.D_e, I_e=m.I_e,
                         tv_state=m.tv_state, tv_state_bound=m.tv_state_bound,
                         tv_sa=m.tv_sa, tv_sa_bound=m.tv_sa_bound,
                         slack_state=m.tv_state_bound - m.tv_state,
                         slack_sa=m.tv_sa_bound - m.tv_sa, violated=0))
    n_viol = sum(r["violated"] for r in rows)
    summary = dict(n_configs=len(rows), n_violated=n_viol,
                   worst_realized_over_bound=worst_ratio)
    return rows, summary


def check_gauge(panel, actor_opt, advisor_seek):
    rows = []
    for beta in (0.30, 0.60, 0.90):
        for t in np.linspace(0.0, 1.0, 11):
            pB_t = (1 - t) * actor_opt + t * advisor_seek   # t=0 agreement -> t=1 full
            m = panel.meter(actor_opt, pB_t, beta)
            rows.append(dict(beta=beta, t=float(t), D_e=m.D_e, I_e=m.I_e,
                             tv_state=m.tv_state, tv_state_bound=m.tv_state_bound))
    mono_ok = True
    for beta in (0.30, 0.60, 0.90):
        sub = [r for r in rows if r["beta"] == beta]
        tv = [r["tv_state"] for r in sub]
        Ie = [r["I_e"] for r in sub]
        if not (np.all(np.diff(Ie) >= -1e-12) and np.all(np.diff(tv) >= -1e-12)):
            mono_ok = False
    summary = dict(monotone_in_disagreement=int(mono_ok),
                   I_e_range=[min(r["I_e"] for r in rows), max(r["I_e"] for r in rows)])
    return rows, summary


def check_gradient(panel, actor, advisor_boltz, eps=1e-5):
    S, A = actor.shape
    rows, worst_b, worst_l = [], 0.0, 0.0
    for beta in BETA_GRID:
        gr = panel.control_gradient(actor, advisor_boltz, beta, tau=TAU)
        Jp = panel.J_ext(Panel.mixture(actor, advisor_boltz, beta + eps))
        Jm = panel.J_ext(Panel.mixture(actor, advisor_boltz, beta - eps))
        fd_b = (Jp - Jm) / (2 * eps)
        sol = panel.solve(Panel.mixture(actor, advisor_boltz, beta))
        AI = sol.A_int
        dpi = (1.0 / TAU) * advisor_boltz * (AI - (advisor_boltz * AI).sum(1, keepdims=True))
        Jp2 = panel.J_ext(Panel.mixture(actor, advisor_boltz + eps * dpi, beta))
        Jm2 = panel.J_ext(Panel.mixture(actor, advisor_boltz - eps * dpi, beta))
        fd_l = (Jp2 - Jm2) / (2 * eps)
        eb, el = abs(fd_b - gr.dJ_dbeta), abs(fd_l - gr.dJ_dlambda)
        worst_b, worst_l = max(worst_b, eb), max(worst_l, el)
        rows.append(dict(beta=beta, G_beta=gr.G_beta, dJ_dbeta=gr.dJ_dbeta, fd_dJ_dbeta=fd_b,
                         err_beta=eb, H_lambda=gr.H_lambda, dJ_dlambda=gr.dJ_dlambda,
                         fd_dJ_dlambda=fd_l, err_lambda=el))
    summary = dict(max_err_beta=worst_b, max_err_lambda=worst_l)
    return rows, summary


def check_sampled(panel, env, actor, advisor, beta):
    S, A = actor.shape
    gamma = panel.gamma
    mix = Panel.mixture(actor, advisor, beta)
    sol = panel.solve(mix)
    d_exact, AE_exact, AI_exact = sol.d, sol.A_ext, sol.A_int
    D_s = panel.statewise_disagreement(actor, advisor)
    m_exact = panel.meter(actor, advisor, beta)

    rows = []
    for n in SAMPLED_BUDGETS:
        roll = collect_rollouts(env, mix, n, seed=SEED)
        d_hat = empirical_occupancy(roll, S, gamma)
        ae_adv, _ = compute_gae(roll, sol.V_ext, "ext", gamma, GAE_LAMBDA)
        ai_adv, _ = compute_gae(roll, sol.V_int, "int", gamma, GAE_LAMBDA)
        ae_hat, _ = bucket_by_sa(roll, ae_adv, S, A)
        ai_hat, _ = bucket_by_sa(roll, ai_adv, S, A)
        De_hat = float(d_hat @ D_s)
        wsa = (d_exact[:, None] * mix)
        ae_rms = float(np.sqrt((wsa * (ae_hat - AE_exact) ** 2).sum() / wsa.sum()))
        ai_rms = float(np.sqrt((wsa * (ai_hat - AI_exact) ** 2).sum() / wsa.sum()))
        trunc = float(roll.trunc.sum() / roll.n_episodes)
        rows.append(dict(budget=n, occ_l1=float(np.abs(d_hat - d_exact).sum()),
                         adv_ext_rms=ae_rms, adv_int_rms=ai_rms,
                         De_exact=m_exact.D_e, De_hat=De_hat, De_err=abs(De_hat - m_exact.D_e),
                         Ie_exact=m_exact.I_e, Ie_hat=beta * De_hat,
                         truncation_rate=trunc))
    summary = dict(beta=beta,
                   occ_l1_final=rows[-1]["occ_l1"], De_err_final=rows[-1]["De_err"],
                   adv_ext_rms_final=rows[-1]["adv_ext_rms"], trunc_final=rows[-1]["truncation_rate"])
    return rows, summary


def check_witness():
    Ahat, dc = 1.0, WITNESS_DC
    pan = single_state_panel(gamma=0.85, A_E=[Ahat, -Ahat])
    piA = np.array([[0.5, 0.5]]); piB = np.array([[0.5, 0.5]])
    piAp = np.array([[0.5 * (1 + dc), 0.5 * (1 - dc)]])
    piBp = np.array([[0.5 * (1 - dc), 0.5 * (1 + dc)]])
    rows, worst, all_admissible = [], 0.0, True
    for beta in WITNESS_BETAS:
        rec = pan.surrogate_recovery(piA, piB, beta, piAp, piBp,
                                     delta_c=dc, delta_phi=DELTA_PHI)
        pred = -2 * beta * dc * Ahat
        err = abs(rec.total - pred)
        rphi_dev = dc * abs(1 - 2 * beta)
        admissible = rphi_dev <= DELTA_PHI + 1e-12
        all_admissible &= admissible
        sat = abs(rec.total) / rec.int_bound if rec.int_bound else float("nan")
        worst = max(worst, err, abs(rec.occ_channel))
        rows.append(dict(beta=beta, delta_c=dc, Ahat=Ahat, occ_channel=rec.occ_channel,
                         int_channel=rec.int_channel, total=rec.total, predicted=pred, err=err,
                         rphi_dev=rphi_dev, delta_phi=DELTA_PHI, admissible=int(admissible),
                         int_bound=rec.int_bound, saturation=sat))
    summary = dict(max_err=worst, all_admissible=int(all_admissible))
    return rows, summary


def main():
    env = GridEnv(size=SIZE, start=START, goal=GOAL, distractors=[(DCELL, R_INT)])
    panel = Panel.from_env(env)
    S, A = env.n_states, env.n_actions
    rng = np.random.default_rng(SEED)
    gamma = panel.gamma

    QE = vi_Q(panel.P, panel.R_ext, gamma)
    QI = vi_Q(panel.P, panel.R_int, gamma)
    actor_opt     = near_greedy(QE, 0.02)
    advisor_seek  = near_greedy(QI, 0.02)
    advisor_boltz = boltzmann(QE + 1.0 * QI, TAU)
    sampled_actor   = near_greedy(QE, 0.02)
    sampled_advisor = near_greedy(QE, 0.10)

    cfg_common = dict(size=SIZE, start=START, goal=GOAL, distractor=DCELL, r_int=R_INT,
                      gamma=gamma, horizon=env.max_steps, tau=TAU,
                      delta_c=DELTA_C, delta_phi=DELTA_PHI, seed=SEED)

    print(f"calibration on {SIZE}x{SIZE}  S={S}  distractor@{DCELL}  gamma={gamma:.4f}  horizon={env.max_steps}\n")

    r1, s1 = check_bounds(panel, actor_opt, advisor_seek, rng)
    save_run(OUTDIR, "calib_bounds", dict(cfg_common, check="occupancy_bounds", n_configs=N_CONFIGS), r1)
    print(f"[1] occupancy bounds: {s1['n_violated']}/{s1['n_configs']} violated; "
          f"worst realized/bound = {s1['worst_realized_over_bound']:.3f}")

    r2, s2 = check_gauge(panel, actor_opt, advisor_seek)
    save_run(OUTDIR, "calib_gauge", dict(cfg_common, check="gauge"), r2)
    print(f"[2] gauge: monotone-in-disagreement={bool(s2['monotone_in_disagreement'])}; "
          f"I_e spans {s2['I_e_range'][0]:.3f}->{s2['I_e_range'][1]:.3f} at fixed/ varied beta")

    r3, s3 = check_gradient(panel, actor_opt, advisor_boltz)
    save_run(OUTDIR, "calib_gradient", dict(cfg_common, check="control_gradient"), r3)
    print(f"[3] control gradient vs FD: max|err| beta={s3['max_err_beta']:.2e}  lambda={s3['max_err_lambda']:.2e}")

    r4, s4 = check_sampled(panel, env, sampled_actor, sampled_advisor, beta=0.30)
    save_run(OUTDIR, "calib_sampled", dict(cfg_common, check="sampled", budgets=list(SAMPLED_BUDGETS)), r4)
    print(f"[4] sampled@{SAMPLED_BUDGETS[-1]}: occ L1={s4['occ_l1_final']:.4f}  "
          f"adv_ext RMS={s4['adv_ext_rms_final']:.4f}  D_e err={s4['De_err_final']:.4f}  "
          f"trunc={s4['trunc_final']:.3f}")

    r5, s5 = check_witness()
    save_run(OUTDIR, "calib_witness", dict(cfg_common, check="integrand_witness", witness_dc=WITNESS_DC), r5)
    print(f"[5] integrand witness: max|total - (-2 beta dc Ahat)| and |occ_channel| = {s5['max_err']:.2e}"
          f"  (trust-region admissible: {bool(s5['all_admissible'])})")

    print("\nall calibration outputs written to", OUTDIR)


if __name__ == "__main__":
    main()