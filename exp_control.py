import numpy as np
from gridworld_env import GridEnv, make_vwall, make_room_field
from panel import Panel
from apparatus import (Apparatus, FeatureMap, boltzmann_advisor, LinearSoftmaxPolicy, advisor_update)
from recording import save_run

SIZE, START, GOAL = 10, (0, 0), (9, 9)
DOOR_ROW = 9
ROOM_PAYOUT = 0.01
D_FEAT = 16
TAU, LR, DELTA_C, DELTA_PHI = 0.30, 8.0, 0.10, 0.05
J_OPT = 0.7278

BETA0 = 0.45
DBUDGET = 0.06
LAM_WITHDRAW_BETA = 0.05
DLAM = 0.01
GBETA_TRIGGER_PATIENCE = 5
LAM_DEFAULT = 1.0
WARMUP_ITERS = 150

SEEDS = list(range(8))
N_ITERS = 10000
TRACE_EVERY = 25
FD_CHECKPOINTS = [0, 100, 250, 500, 1000, 2000, 3500, 5000, 7000, N_ITERS - 1]
FD_EPS_B = 1e-3
FD_EPS_L = 1e-3
FD_K = 300
OUTDIR = "runs/expC"
RUN_NAME = "expC_control"


def build():
    walls = make_vwall(SIZE, 4, DOOR_ROW)
    field = make_room_field(SIZE, ROOM_PAYOUT, col_min=5, exclude=[GOAL])
    env = GridEnv(SIZE, START, GOAL, wall_edges=walls, intrinsic_field=field)
    panel = Panel.from_env(env)
    room_states = np.array([s for s in range(SIZE * SIZE) if (s % SIZE) >= 5])
    return env, panel, room_states


def find_lambda_ceiling(panel, pi_A0, beta, tau, lam_grid):
    Hs = []
    for lam in lam_grid:
        pi_B, _ = boltzmann_advisor(panel, pi_A0, beta, lam, tau)
        gr = panel.control_gradient(pi_A0, pi_B, beta, tau=tau)
        Hs.append(gr.H_lambda)
    Hs = np.array(Hs)
    for i in range(len(lam_grid) - 1):
        if Hs[i] >= 0.0 >= Hs[i + 1]:
            t = Hs[i] / (Hs[i] - Hs[i + 1] + 1e-18)
            return float(lam_grid[i] + t * (lam_grid[i + 1] - lam_grid[i])), Hs
    return float(lam_grid[-1]), Hs


def _deployed_advisor_probs(panel, ap, beta, lam, tau):
    if ap.mode == "boltzmann":
        return boltzmann_advisor(panel, ap.polA.probs(), beta, lam, tau)[0]
    return ap.polB.probs()


def shaped_lambda_fd(panel, ap, beta, lam, *, eps_l=FD_EPS_L, K=FD_K):
    pi_A = ap.polA.probs()

    def reequilibrate(lam_t):
        pol = LinearSoftmaxPolicy(ap.fm, ap.A)
        pol.W = ap.polB.W.copy()
        for _ in range(K):
            pi_B = pol.probs()
            sol = panel.solve(panel.mixture(pi_A, pi_B, beta))
            A_B = sol.A_ext + lam_t * sol.A_int
            advisor_update(pol, pi_A, beta, A_B, sol.d,
                           lr=ap.lr, delta_c=ap.delta_c, delta_phi=ap.delta_phi)
        return pol.probs()

    l_hi, l_lo = lam + eps_l, max(0.0, lam - eps_l)
    pBp = reequilibrate(l_hi)
    pBm = reequilibrate(l_lo)
    Jlp = panel.J_ext(panel.mixture(pi_A, pBp, beta))
    Jlm = panel.J_ext(panel.mixture(pi_A, pBm, beta))
    span = l_hi - l_lo
    return (Jlp - Jlm) / span if span > 0 else float("nan")


def fd_dial_check(panel, ap, beta, lam, tau, *, eps_b=FD_EPS_B, eps_l=FD_EPS_L, K=FD_K):
    pi_A = ap.polA.probs()
    pi_B = _deployed_advisor_probs(panel, ap, beta, lam, tau)
    gr = panel.control_gradient(pi_A, pi_B, beta, tau=tau)
    b_hi, b_lo = min(1.0, beta + eps_b), max(0.0, beta - eps_b)
    Jbp = panel.J_ext(panel.mixture(pi_A, pi_B, b_hi))
    Jbm = panel.J_ext(panel.mixture(pi_A, pi_B, b_lo))
    beta_real = (Jbp - Jbm) / (b_hi - b_lo)
    beta_pred = gr.dJ_dbeta
    lam_pred = gr.dJ_dlambda
    if ap.mode == "boltzmann":
        l_hi, l_lo = lam + eps_l, max(0.0, lam - eps_l)
        pBp = boltzmann_advisor(panel, pi_A, beta, l_hi, tau)[0]
        pBm = boltzmann_advisor(panel, pi_A, beta, l_lo, tau)[0]
        Jlp = panel.J_ext(panel.mixture(pi_A, pBp, beta))
        Jlm = panel.J_ext(panel.mixture(pi_A, pBm, beta))
        lam_real = (Jlp - Jlm) / (l_hi - l_lo) if (l_hi - l_lo) > 0 else float("nan")
    else:
        lam_real = float("nan")
    return beta_pred, beta_real, lam_pred, lam_real


def check_K_convergence(panel, ap, beta, lam, *, eps_l=FD_EPS_L, K_grid=(50, 100, 200, 300, 500)):
    if ap.mode != "shaped":
        return []
    return [(K, shaped_lambda_fd(panel, ap, beta, lam, eps_l=eps_l, K=K)) for K in K_grid]


def run(n_iters=N_ITERS, seed=0, advisor_mode="boltzmann", gamma=None, checkpoints=None, verbose=True, fd_K=FD_K, warmup_iters=WARMUP_ITERS, lam_default=LAM_DEFAULT):
    env, panel, room_states = build()
    g = env.gamma if gamma is None else gamma
    S, A = env.n_states, env.n_actions
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    pi_A0 = np.full((S, A), 1.0 / A)

    if advisor_mode == "shaped":
        lam_star = lam_default
        warm = int(warmup_iters)
    else:
        lam_star, _ = find_lambda_ceiling(panel, pi_A0, BETA0, TAU,[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        warm = 0
        if verbose:
            print(f"[find] lambda* (Boltzmann ceiling at initial actor) = {lam_star:.3f}")

    ap = Apparatus(panel, fm, A, advisor_mode=advisor_mode, beta=BETA0, lam=lam_star,
                   tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                   distractor_states=room_states, seed=seed)
    solo = Apparatus(panel, fm, A, advisor_mode="none", beta=0.0,
                     tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI, seed=seed)

    if checkpoints is None:
        checkpoints = sorted(set([c for c in FD_CHECKPOINTS if c < n_iters] + [n_iters - 1]))
    beta, lam = BETA0, lam_star
    withdrawing = False
    g_pos_streak = 0
    fd_rows = []
    traj = []
    warm_H = float("nan")
    warm_confirmed = None

    for e in range(n_iters):
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = ap.step(beta=beta, lam=lam)
        solo.step()
        G, D = rec["G_beta"], rec["D_e"]

        if advisor_mode == "shaped" and e == warm - 1:
            warm_H = panel.control_gradient(ap.polA.probs(), ap.polB.probs(),
                                            beta, tau=TAU).H_lambda
            warm_confirmed = bool(warm_H > 0)
            if verbose:
                tag = "useful (H>0)" if warm_confirmed else "OVER-SHAPING (H<=0)"
                print(f"[warmup] trained-guide H_lambda at lam={lam:.2f}, {warm} iters "
                      f"= {warm_H:+.6f} -> {tag}; holding lam={lam_star:.2f}")

        if e >= warm:
            if not withdrawing:
                g_pos_streak = g_pos_streak + 1 if G > 0 else 0
                if g_pos_streak >= GBETA_TRIGGER_PATIENCE:
                    withdrawing = True
            if withdrawing:
                dbeta = (1.0 - g) / g * DBUDGET / max(D, 1e-6)
                beta = max(0.0, beta - dbeta)
                if beta < LAM_WITHDRAW_BETA:
                    lam = max(0.0, lam - DLAM)

        traj.append((e, beta, lam, rec["J_ext_actor"], rec["J_ext_mix"],
                     rec["J_ext_advisor"], G, rec["H_lambda"], rec["dJ_dbeta"],
                     rec["dJ_dlambda"], D, rec["I_e"], rec["distr_occ_mix"],
                     solo.log[-1]["J_ext_actor"]))

        if e in checkpoints:
            bp, br, lp, lr_ = fd_dial_check(panel, ap, beta, lam, TAU, K=fd_K)
            fd_rows.append((e, bp, br, lp, lr_))

    return dict(lam_star=lam_star, traj=traj, fd=fd_rows, ap=ap, solo=solo,
                panel=panel, gamma=g, advisor_mode=advisor_mode,
                warm_H=warm_H, warm_confirmed=warm_confirmed)


def report(res):
    g = res["gamma"]
    print("\n=== trajectory (J normalized by J_opt) ===")
    print(" iter |  beta   lam  | Jact  Jmix  Jadv | G_beta   H_lam  | roomOcc | Jsolo")
    keep = [0, 100, 250, 500, 1000, 1500, 2000, len(res["traj"]) - 1]
    for row in res["traj"]:
        (e, beta, lam, Ja, Jm, Jb, G, H, dJb, dJl, D, Ie, room, Js) = row
        if e in keep:
            print(f"{e:5d} | {beta:.3f} {lam:.3f} | {Ja/J_OPT:.3f} {Jm/J_OPT:.3f} "
                  f"{Jb/J_OPT:.3f} | {G:+.4f} {H:+.5f} | {room:.3f} | {Js/J_OPT:.3f}")
    print("\n=== dial accuracy on the learned trajectory (predicted vs realized dJ^E) ===")
    print(" iter | beta_pred  beta_real  (ratio) | lam_pred   lam_real   (ratio)")
    for (e, bp, br, lp, lr_) in res["fd"]:
        rb = br / bp if abs(bp) > 1e-9 else float("nan")
        rl = lr_ / lp if abs(lp) > 1e-9 else float("nan")
        print(f"{e:5d} | {bp:+.5f}  {br:+.5f}  ({rb:5.2f}) | {lp:+.6f}  {lr_:+.6f}  ({rl:5.2f})")
    end = res["traj"][-1]
    solo_end = end[-1]
    print("\n=== terminus / handoff ===")
    print(f"final beta={end[1]:.4f}  lambda={end[2]:.4f}")
    print(f"actor J^E/J_opt = {end[3]/J_OPT:.4f}   solo J^E/J_opt = {solo_end/J_OPT:.4f}"
          f"   residual gap = {(solo_end-end[3])/J_OPT:+.4f}")


def run_all(advisor_mode="boltzmann", seeds=None, n_iters=None, trace_every=None, outdir=None, name=None, fd_K=None):
    import time
    seeds = SEEDS if seeds is None else list(seeds)
    n_iters = N_ITERS if n_iters is None else int(n_iters)
    trace_every = TRACE_EVERY if trace_every is None else int(trace_every)
    outdir = OUTDIR if outdir is None else outdir
    name = (f"{RUN_NAME}_{advisor_mode}") if name is None else name
    fd_K = FD_K if fd_K is None else int(fd_K)

    t0 = time.time()
    per_seed = []
    print(f"Exp C control process [{advisor_mode}]: seeds={seeds}  n_iters={n_iters}  "
          f"door_row={DOOR_ROW}  c={ROOM_PAYOUT}  beta0={BETA0}\n")
    for s in seeds:
        res = run(n_iters=n_iters, seed=s, advisor_mode=advisor_mode, verbose=False, fd_K=fd_K)
        per_seed.append(res)
        end = res["traj"][-1]
        wtag = ""
        if advisor_mode == "shaped":
            wtag = f"  warmH={res['warm_H']:+.5f}({'ok' if res['warm_confirmed'] else 'OVER'})"
        print(f"[seed {s}] lam*={res['lam_star']:.3f}  final beta={end[1]:.3f} "
              f"lam={end[2]:.3f}  actor={end[3]/J_OPT:.3f}  solo={end[13]/J_OPT:.3f} "
              f"gap={(end[13]-end[3])/J_OPT:+.4f}{wtag}  [{time.time()-t0:.0f}s]")

    n = len(seeds)
    full_iters = [row[0] for row in per_seed[0]["traj"]]
    tidx = [i for i, e in enumerate(full_iters)
            if (e % trace_every == 0) or (e == full_iters[-1])]
    trace_iters = np.array([full_iters[i] for i in tidx])

    def col(j):
        return np.array([[per_seed[si]["traj"][i][j] for i in tidx] for si in range(n)])

    arrays = dict(
        seeds=np.array(seeds), trace_iters=trace_iters,
        beta=col(1), lam=col(2), J_actor=col(3), J_mix=col(4), J_advisor=col(5),
        G_beta=col(6), H_lambda=col(7), dJ_dbeta=col(8), dJ_dlambda=col(9),
        D_e=col(10), I_e=col(11), room_occ=col(12), J_solo=col(13),
        lam_star=np.array([r["lam_star"] for r in per_seed]),
    )
    fd_iters = np.array([row[0] for row in per_seed[0]["fd"]])

    def fdcol(j):
        return np.array([[r["fd"][k][j] for k in range(len(fd_iters))] for r in per_seed])

    arrays.update(fd_iters=fd_iters, fd_beta_pred=fdcol(1), fd_beta_real=fdcol(2),
                  fd_lam_pred=fdcol(3), fd_lam_real=fdcol(4),
                  final_beta=arrays["beta"][:, -1], final_lam=arrays["lam"][:, -1],
                  final_J_actor=arrays["J_actor"][:, -1], final_J_solo=arrays["J_solo"][:, -1],
                  warm_H=np.array([r["warm_H"] for r in per_seed]))

    log = []
    for si, s in enumerate(seeds):
        for i in tidx:
            r = per_seed[si]["traj"][i]
            log.append(dict(seed=int(s), iter=int(r[0]), beta=r[1], lam=r[2],
                            J_ext_actor=r[3], J_ext_mix=r[4], J_ext_advisor=r[5],
                            J_ext_solo=r[13], G_beta=r[6], H_lambda=r[7],
                            dJ_dbeta=r[8], dJ_dlambda=r[9], D_e=r[10], I_e=r[11],
                            room_occ=r[12]))

    config = dict(size=SIZE, start=list(START), goal=list(GOAL), door_row=DOOR_ROW,
                  room_payout=ROOM_PAYOUT, d_feat=D_FEAT, tau=TAU, lr=LR,
                  delta_c=DELTA_C, delta_phi=DELTA_PHI, beta0=BETA0, dbudget=DBUDGET,
                  lam_withdraw_beta=LAM_WITHDRAW_BETA, dlam=DLAM,
                  gbeta_patience=GBETA_TRIGGER_PATIENCE, seeds=list(seeds),
                  n_iters=n_iters, trace_every=trace_every,
                  fd_checkpoints=[int(x) for x in fd_iters],
                  advisor_mode=advisor_mode, fd_eps_b=FD_EPS_B, fd_eps_l=FD_EPS_L, fd_K=fd_K,
                  lam_default=LAM_DEFAULT, warmup_iters=WARMUP_ITERS,
                  j_opt=J_OPT, gamma=float(per_seed[0]["gamma"]))
    paths = save_run(outdir, name, config, log, arrays=arrays)
    report_pooled(arrays, advisor_mode=advisor_mode)
    print(f"\nwrote {n} seeds in {time.time()-t0:.0f}s:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return arrays, paths


def report_pooled(A, advisor_mode="boltzmann"):
    it = A["trace_iters"]
    ck = [c for c in (0, 100, 250, 500, 1000, 2000, 3500, 5000, 7000) if c <= it[-1]] + [int(it[-1])]
    print("\n=== pooled trajectory (mean+/-std over seeds, J/J_opt) ===")
    print(" iter |   beta            lam        |  J_actor        J_mix          J_solo")
    for c in ck:
        i = int(np.argmin(np.abs(it - c)))
        def ms(name, norm=1.0):
            v = A[name][:, i] / norm
            return f"{v.mean():.3f}+/-{v.std():.3f}"
        print(f"{int(it[i]):5d} | {ms('beta')}  {ms('lam')} | "
              f"{ms('J_actor', J_OPT)}  {ms('J_mix', J_OPT)}  {ms('J_solo', J_OPT)}")

    print(f"\n=== pooled dial accuracy [{advisor_mode}] (mean over seeds: predicted vs realized dJ^E) ===")
    fi = A["fd_iters"]
    if advisor_mode == "boltzmann":
        print(" beta dial exact against the deployed guide (ratio -> 1, the control axis);")
        print(" lambda re-solve FD is leading-order -- the fixed-Q model drops d_lambda Q,")
        print(" so ratio ~1 with the Q-response visible (magnitude well-defined, unlike the")
        print(" trained guide). Formula-exactness (fixed-Q, ratio->1) is lambda_formula_check.")
        print(" iter | beta_pred   beta_real   (ratio) | lam_pred    lam_real    (ratio)")
        for k in range(len(fi)):
            bp, br = A["fd_beta_pred"][:, k].mean(), A["fd_beta_real"][:, k].mean()
            lp, lr_ = A["fd_lam_pred"][:, k].mean(), A["fd_lam_real"][:, k].mean()
            rb = br / bp if abs(bp) > 1e-9 else float("nan")
            rl = lr_ / lp if abs(lp) > 1e-9 else float("nan")
            print(f"{int(fi[k]):5d} | {bp:+.5f}   {br:+.5f}   ({rb:5.2f}) | "
                  f"{lp:+.6f}  {lr_:+.6f}  ({rl:5.2f})")
    else:
        print(" beta dial exact + quantitative (ratio -> 1, the control axis); lambda")
        print(" magnitude not differenced (trained-guide ill-posed) -- axis is the H_lambda")
        print(" meter below; sign-faithfulness validated by the kcheck diagnostic.")
        print(" iter | beta_pred   beta_real   (ratio) | lam_pred (= meter sign)")
        for k in range(len(fi)):
            bp, br = A["fd_beta_pred"][:, k].mean(), A["fd_beta_real"][:, k].mean()
            lp = A["fd_lam_pred"][:, k].mean()
            rb = br / bp if abs(bp) > 1e-9 else float("nan")
            print(f"{int(fi[k]):5d} | {bp:+.5f}   {br:+.5f}   ({rb:5.2f}) | "
                  f"{lp:+.6f}  ({'+' if lp > 0 else '-'})")

        print("\n=== over-shaping onset: trained-guide H_lambda meter (mean+/-std) ===")
        H = A["H_lambda"]
        Hm = H.mean(axis=0)
        cross = next((int(it[i]) for i in range(len(it)) if Hm[i] <= 0), None)
        for c in ck:
            i = int(np.argmin(np.abs(it - c)))
            print(f"{int(it[i]):5d} | H_lambda = {H[:, i].mean():+.6f}+/-{H[:, i].std():.6f}"
                  f"  ({'+' if Hm[i] > 0 else '-'})")
        print(f"mean H_lambda sign-flip (onset) near iter {cross}" if cross is not None
              else "mean H_lambda stays positive over the trace")
        if "warm_H" in A:
            wH = A["warm_H"]
            n_ok = int(np.sum(wH > 0))
            print(f"warmup default-confirm: H_lambda(lam={LAM_DEFAULT}) > 0 on {n_ok}/{len(wH)} "
                  f"seeds (mean {wH.mean():+.6f})")

    print("\n=== pooled terminus / handoff ===")
    fb, fl = A["final_beta"], A["final_lam"]
    fa, fs = A["final_J_actor"] / J_OPT, A["final_J_solo"] / J_OPT
    gap = fs - fa
    lam_label = ("lambda* (Boltzmann ceiling)" if advisor_mode == "boltzmann"
                 else "lambda held (warmup-confirmed default)")
    print(f"{lam_label} = {A['lam_star'].mean():.3f}+/-{A['lam_star'].std():.3f}")
    print(f"final beta = {fb.mean():.4f}+/-{fb.std():.4f}   lambda = {fl.mean():.4f}+/-{fl.std():.4f}")
    print(f"actor J/J_opt = {fa.mean():.4f}+/-{fa.std():.4f}   solo = {fs.mean():.4f}+/-{fs.std():.4f}"
          f"   residual gap = {gap.mean():+.4f}+/-{gap.std():.4f}")


def kcheck(seed=0, hold_iters=2000, beta=None, lam=None):
    env, panel, room_states = build()
    S, A = env.n_states, env.n_actions
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    pi_A0 = np.full((S, A), 1.0 / A)
    beta = BETA0 if beta is None else float(beta)
    if lam is None:
        lam, _ = find_lambda_ceiling(panel, pi_A0, beta, TAU, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap = Apparatus(panel, fm, A, advisor_mode="shaped", beta=beta, lam=lam,
                   tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                   distractor_states=room_states, seed=seed)
    for _ in range(hold_iters):
        ap.step(beta=beta, lam=lam)
    pred = panel.control_gradient(ap.polA.probs(), ap.polB.probs(), beta, tau=TAU).dJ_dlambda
    rows = check_K_convergence(panel, ap, beta, lam)
    print(f"\n=== shaped lambda-FD K-convergence (beta={beta} held, lam={lam:.3f}, "
          f"{hold_iters} iters) ===")
    print(f"  dial dJ_dlambda (pred) = {pred:+.6f}")
    for K, v in rows:
        print(f"  K={K:4d}: lam_real={v:+.6f}   ratio={v/pred if abs(pred)>1e-9 else float('nan'):+.3f}")
    return rows


def lambda_formula_check(seed=0, hold_iters=1000, beta=None, lam=None, eps=1e-3):
    env, panel, room = build()
    S, A = env.n_states, env.n_actions
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    pi_A0 = np.full((S, A), 1.0 / A)
    beta = BETA0 if beta is None else float(beta)
    if lam is None:
        lam, _ = find_lambda_ceiling(panel, pi_A0, beta, TAU, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap = Apparatus(panel, fm, A, advisor_mode="boltzmann", beta=beta, lam=lam,
                   tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                   distractor_states=room, seed=seed)
    for _ in range(hold_iters):
        ap.step(beta=beta, lam=lam)
    pi_A = ap.polA.probs()
    pi_B = boltzmann_advisor(panel, pi_A, beta, lam, TAU)[0]
    gr = panel.control_gradient(pi_A, pi_B, beta, tau=TAU)
    sol = panel.solve(panel.mixture(pi_A, pi_B, beta))
    AE, AI = sol.A_ext, sol.A_int

    def fixed_q(lt):
        z = (AE + lt * AI) / TAU
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    Jp = panel.J_ext(panel.mixture(pi_A, fixed_q(lam + eps), beta))
    Jm = panel.J_ext(panel.mixture(pi_A, fixed_q(lam - eps), beta))
    fd = (Jp - Jm) / (2 * eps)
    r = fd / gr.dJ_dlambda if abs(gr.dJ_dlambda) > 1e-12 else float("nan")
    print(f"\n=== lambda-dial formula check (fixed-Q; beta={beta}, lam={lam:.3f}, "
          f"{hold_iters} iters) ===")
    print(f"  dial dJ_dlambda = {gr.dJ_dlambda:+.8f}")
    print(f"  fixed-Q FD      = {fd:+.8f}   ratio = {r:.6f}  (formula exact when ~1)")
    return gr.dJ_dlambda, fd


def run_expC(seeds=None, n_iters=None, fd_K=None):
    out = {}
    for mode in ("shaped", "boltzmann"):
        print(f"\n########## Exp C controller :: advisor_mode={mode} ##########")
        out[mode] = run_all(advisor_mode=mode, seeds=seeds, n_iters=n_iters, fd_K=fd_K)
    return out


if __name__ == "__main__":
    lambda_formula_check()
    kcheck()
    run_expC()