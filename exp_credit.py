from __future__ import annotations
import time
import numpy as np
from gridworld_env import GridEnv
from panel import Panel
from apparatus import Apparatus, FeatureMap
from recording import save_run

SIZE, START, GOAL, DCELL, R_INT = 10, (0, 0), (9, 9), (9, 0), 0.04
D = 16
BETA_FIXED = 0.45
BETA_GRID = [0.0, 0.1, 0.25, 0.45, 0.6, 0.8]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
LR, DELTA_C, DELTA_PHI, TAU = 8.0, 0.10, 0.05, 0.30
SEEDS = range(8)
N_ITERS = 10000
TRACE_EVERY = 100
OUTDIR, NAME = "runs/expA", "expA_credit"


def value_iteration_J(P, R_ext, gamma, mu0, iters=4000, tol=1e-13):
    S, A, _ = P.shape
    R = np.asarray(R_ext)
    Q = np.zeros((S, A))
    for _ in range(iters):
        V = Q.max(1)
        Qn = R + gamma * np.einsum("sap,p->sa", P, V)
        if np.max(np.abs(Qn - Q)) < tol:
            Q = Qn
            break
        Q = Qn
    pi = np.zeros((S, A))
    pi[np.arange(S), Q.argmax(1)] = 1.0
    return float(mu0 @ np.linalg.solve(np.eye(S) - gamma * np.einsum("sa,sap->sp", pi, P), (pi * R).sum(1)))


def exact_blended(panel, didx, lam, iters=4000, tol=1e-13):
    P, Rext, Rint, g = panel.P, np.asarray(panel.R_ext), np.asarray(panel.R_int), panel.gamma
    S, A, _ = P.shape
    R = Rext + lam * Rint
    Q = np.zeros((S, A))
    for _ in range(iters):
        V = Q.max(1)
        Qn = R + g * np.einsum("sap,p->sa", P, V)
        if np.max(np.abs(Qn - Q)) < tol:
            Q = Qn
            break
        Q = Qn
    pi = np.zeros((S, A))
    pi[np.arange(S), Q.argmax(1)] = 1.0
    sol = panel.solve(pi)
    return float(sol.J_ext), float(sol.J_int), float(sol.d[didx])


_TRACE_KEYS = ("J_ext_actor", "J_ext_mix", "J_ext_advisor", "J_ext_rb",
               "distr_occ_actor", "distr_occ_mix", "distr_occ_advisor", "distr_occ_rb",
               "D_e", "I_e", "G_beta", "dJ_dbeta", "H_lambda", "dJ_dlambda")
_TRACE_COLUMNS = ("iter",) + _TRACE_KEYS


def run_config(panel, A, didx, J_opt, *, mode, beta, lam, with_rb, seed):
    S = panel.P.shape[0]
    fm = FeatureMap.random(S, D, seed=seed)
    ap = Apparatus(panel, fm, A, advisor_mode=mode, beta=beta, lam=lam, tau=TAU,
                   lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI, with_rb=with_rb,
                   distractor_states=[didx], seed=seed)
    trace = []
    for t in range(N_ITERS):
        rec = ap.step()
        if t == 0 or (t + 1) % TRACE_EVERY == 0:
            trace.append([t + 1] + [float(rec[k]) if rec.get(k) is not None else np.nan
                                    for k in _TRACE_KEYS])
    last = ap.log[-1]

    piA = ap.polA.probs()
    piB = ap.polB.probs() if ap.polB is not None else piA
    piR = ap.polR.probs() if ap.polR is not None else None
    eff_beta = 0.0 if mode == "none" else beta
    pi_phi = Panel.mixture(piA, piB, eff_beta)

    solA, solB, solPhi = panel.solve(piA), panel.solve(piB), panel.solve(pi_phi)
    m = panel.meter(piA, piB, eff_beta)
    gr = panel.control_gradient(piA, piB, eff_beta, tau=TAU)


    rec = {
        "arm": None, "knob": None, "knob_value": None, "seed": seed,
        "d": D, "mode": mode, "beta": eff_beta, "lam": lam,
        "J_actor": solA.J_ext, "Jint_actor": solA.J_int, "occ_actor": float(solA.d[didx]),
        "J_advisor": solB.J_ext, "Jint_advisor": solB.J_int, "occ_advisor": float(solB.d[didx]),
        "J_mix": solPhi.J_ext, "Jint_mix": solPhi.J_int, "occ_mix": float(solPhi.d[didx]),
        "D_e": m.D_e, "I_e": m.I_e,
        "tv_state": m.tv_state, "tv_state_bound": m.tv_state_bound,
        "tv_sa": m.tv_sa, "tv_sa_bound": m.tv_sa_bound,
        "G_beta": gr.G_beta, "dJ_dbeta": gr.dJ_dbeta, "delta_wd": gr.delta_wd,
        "H_lambda": gr.H_lambda, "dJ_dlambda": gr.dJ_dlambda,
        "A_max": gr.A_max, "G_beta_bound": gr.G_beta_bound,
        "H_lambda_cs_bound": gr.H_lambda_cs_bound,
        "max_rA_dev": last.get("max_rA_dev"), "max_rB_dev": last.get("max_rB_dev"),
        "max_rphi_dev": last.get("max_rphi_dev"),
    }
    finals = {"piA": piA, "piB": piB}
    if piR is not None:
        solR = panel.solve(piR)
        rec.update({"J_rb": solR.J_ext, "Jint_rb": solR.J_int, "occ_rb": float(solR.d[didx])})
        finals["piR"] = piR
    for k in ("J_actor", "J_mix", "J_advisor", "J_rb"):
        rec[k.replace("J_", "pct_")] = (rec[k] / J_opt) if rec.get(k) is not None else None
    return rec, np.asarray(trace, dtype=np.float64), finals


def main():
    env = GridEnv(size=SIZE, start=START, goal=GOAL, distractors=[(DCELL, R_INT)])
    panel = Panel.from_env(env)
    S, A = env.n_states, env.n_actions
    didx = int(np.argwhere(np.asarray(panel.R_int).sum(1) > 0)[0, 0])
    J_opt = value_iteration_J(panel.P, panel.R_ext, panel.gamma, panel.mu0)

    print(f"Experiment A -- credit isolation.  {SIZE}x{SIZE} S={S}  trap@{DCELL}(idx {didx})"
          f"  d={D}  J_opt={J_opt:.4f}")
    print(f"lam grid {LAM_GRID}   beta grid {BETA_GRID}   seeds {list(SEEDS)}"
          f"   {N_ITERS} iters\n")

    log, arrays = [], {}
    t0 = time.time()

    def store(tag, tr, finals):
        arrays[f"trace_{tag}"] = tr
        for pname, parr in finals.items():
            arrays[f"final_{pname}_{tag}"] = parr

    print("exact blended-optimal anchor (value iteration):")
    for lam in LAM_GRID:
        Je, Ji, occ = exact_blended(panel, didx, lam)
        log.append({"arm": "exact", "knob": "lambda", "knob_value": lam, "seed": -1,
                    "d": D, "mode": "vi", "beta": 0.0, "lam": lam,
                    "J_rb": Je, "Jint_rb": Ji, "occ_rb": occ, "pct_rb": Je / J_opt})
        print(f"  lam={lam:.2f}  J_rb={100*Je/J_opt:4.0f}%  occ_rb={occ:.3f}")

    print("\narm rb  (foil credit-open, lam sweep, mode none):")
    for lam in LAM_GRID:
        for s in SEEDS:
            rec, tr, finals = run_config(panel, A, didx, J_opt,
                                         mode="none", beta=0.0, lam=lam, with_rb=True, seed=s)
            rec.update({"arm": "rb", "knob": "lambda", "knob_value": lam})
            log.append(rec)
            store(f"rb_lam{lam}_s{s}", tr, finals)
        dr = [r for r in log if r["arm"] == "rb" and r["knob_value"] == lam]
        print(f"  lam={lam:.2f}  occ_rb={np.mean([r['occ_rb'] for r in dr]):.3f}"
              f"  pct_rb={100*np.mean([r['pct_rb'] for r in dr]):4.0f}%"
              f"  | occ_actor={np.mean([r['occ_actor'] for r in dr]):.3f}"
              f"  pct_actor={100*np.mean([r['pct_actor'] for r in dr]):4.0f}%"
              f"   [{time.time()-t0:5.0f}s]")

    print("\narm disjoint  (BB headline, pure-intrinsic advisor, beta sweep):")
    for beta in BETA_GRID:
        mode = "none" if beta == 0.0 else "disjoint"
        for s in SEEDS:
            rec, tr, finals = run_config(panel, A, didx, J_opt, mode=mode, beta=beta, lam=1.0, with_rb=False, seed=s)
            rec.update({"arm": "disjoint", "knob": "beta", "knob_value": beta})
            log.append(rec)
            store(f"disjoint_beta{beta}_s{s}", tr, finals)
        dr = [r for r in log if r["arm"] == "disjoint" and r["knob_value"] == beta]
        print(f"  beta={beta:.2f}  occ_actor={np.mean([r['occ_actor'] for r in dr]):.3f}"
              f"  pct_actor={100*np.mean([r['pct_actor'] for r in dr]):4.0f}%"
              f"  | occ_mix={np.mean([r['occ_mix'] for r in dr]):.3f}"
              f"  occ_adv={np.mean([r['occ_advisor'] for r in dr]):.3f}"
              f"  I_e={np.mean([r['I_e'] for r in dr]):.3f}   [{time.time()-t0:5.0f}s]")

    print(f"\narm shaped  (matched variant, advisor A^E+lam A^I, beta={BETA_FIXED}, lam sweep):")
    for lam in LAM_GRID:
        for s in SEEDS:
            rec, tr, finals = run_config(panel, A, didx, J_opt,
                                         mode="shaped", beta=BETA_FIXED, lam=lam, with_rb=False, seed=s)
            rec.update({"arm": "shaped", "knob": "lambda", "knob_value": lam})
            log.append(rec)
            store(f"shaped_lam{lam}_s{s}", tr, finals)
        dr = [r for r in log if r["arm"] == "shaped" and r["knob_value"] == lam]
        print(f"  lam={lam:.2f}  occ_actor={np.mean([r['occ_actor'] for r in dr]):.3f}"
              f"  pct_actor={100*np.mean([r['pct_actor'] for r in dr]):4.0f}%"
              f"  | occ_mix={np.mean([r['occ_mix'] for r in dr]):.3f}"
              f"  occ_adv={np.mean([r['occ_advisor'] for r in dr]):.3f}   [{time.time()-t0:5.0f}s]")

    config = dict(
        experiment="expA_credit", size=SIZE, start=START, goal=GOAL,
        distractor=DCELL, r_int=R_INT, distractor_idx=didx,
        d=D, beta_fixed=BETA_FIXED, beta_grid=BETA_GRID, lam_grid=LAM_GRID,
        lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI, tau=TAU,
        gamma=panel.gamma, horizon=env.max_steps, J_opt=J_opt,
        seeds=list(SEEDS), n_iters=N_ITERS, trace_every=TRACE_EVERY,
        trace_columns=list(_TRACE_COLUMNS),
        array_keys=("trace_<tag> = (n_checkpoints, len(trace_columns)) learning curve; "
                    "final_{piA,piB,piR}_<tag> = (S,A) converged policy tables; "
                    "tag = <arm>_<knob><value>_s<seed>. Any panel observable can be "
                    "recomputed offline from a final policy + the panel built from config."),
    )
    paths = save_run(OUTDIR, NAME, config, log, arrays=arrays)
    n_tr = sum(k.startswith("trace_") for k in arrays)
    n_fin = sum(k.startswith("final_") for k in arrays)
    print(f"\nwrote {len(log)} rows, {n_tr} traces, {n_fin} final-policy arrays to {OUTDIR}")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()