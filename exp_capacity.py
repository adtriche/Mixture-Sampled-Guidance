import time
import numpy as np
from gridworld_env import GridEnv
from panel import Panel
from apparatus import FeatureMap, Apparatus
from recording import save_run

SIZE, START, GOAL, DCELL, R_INT = 10, (0, 0), (9, 9), (9, 0), 0.04

BETA = 0.45
LR = 8.0
DELTA_C = 0.10
DELTA_PHI = 0.05
TAU = 0.30

D_GRID = [4, 6, 8, 12, 16, 24, 40, 64, 100]
SEEDS = list(range(8))
N_ITERS = 10000
TRACE_EVERY = 200

TAU_RESID = 0.05

OUTDIR = "runs/expB"
SAVE_TRACES = True


def value_iteration_J(P, R, gamma, mu0, iters=8000, tol=1e-11):
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(iters):
        Q = R + gamma * np.einsum("sax,x->sa", P, V)
        Vn = Q.max(1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    return float(mu0 @ V)


def train_arm(panel, S, A, d, seed, mode, beta, distractor_states):
    fm = FeatureMap.random(S, d, seed=seed)
    app = Apparatus(
        panel, fm, A,
        advisor_mode=mode, beta=beta, lam=1.0, tau=TAU,
        lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
        distractor_states=distractor_states, seed=seed,
    )
    trace = []
    for it in range(N_ITERS):
        app.step()
        if it % TRACE_EVERY == 0 or it == N_ITERS - 1:
            trace.append((it, app.log[-1]["J_ext_actor"]))
    return app.log[-1], np.asarray(trace, dtype=float)


def main():
    env = GridEnv(size=SIZE, start=START, goal=GOAL, distractors=[(DCELL, R_INT)])
    panel = Panel.from_env(env)
    S, A = env.n_states, env.n_actions
    didx = int(np.argwhere(np.asarray(panel.R_int).sum(1) > 0)[0, 0])
    supp = [didx]
    J_opt = value_iteration_J(panel.P, panel.R_ext, panel.gamma, panel.mu0)

    print(f"env {SIZE}x{SIZE}  S={S}  start{START} goal{GOAL}  distractor@{DCELL}(idx {didx})"
          f"  r_int={R_INT}")
    print(f"gamma={panel.gamma:.4f}  horizon={env.max_steps}  J_opt={J_opt:.4f}")
    print(f"sweep: d={D_GRID}  seeds={list(SEEDS)}  n_iters={N_ITERS}  beta={BETA}\n")

    rows, arrays = [], {}
    t_start = time.time()
    for d in D_GRID:
        for s in SEEDS:
            cl, tr_cl = train_arm(panel, S, A, d, s, "none",     0.0,  supp)
            mx, tr_mx = train_arm(panel, S, A, d, s, "disjoint", BETA, supp)
            Jc, Jm = cl["J_ext_actor"], mx["J_ext_actor"]
            rows.append({
                "d": d, "seed": s,
                "J_clean": Jc, "J_mix": Jm, "residual": Jc - Jm,
                "pct_clean": Jc / J_opt, "pct_mix": Jm / J_opt,
                "occ_clean": cl["distr_occ_actor"], "occ_mix": mx["distr_occ_actor"],
                "J_deployed_mix": mx["J_ext_mix"], "pct_deployed_mix": mx["J_ext_mix"] / J_opt,
                "occ_deployed_mix": mx["distr_occ_mix"], "J_advisor": mx["J_ext_advisor"],
                "occ_advisor": mx["distr_occ_advisor"],
            })
            if SAVE_TRACES:
                arrays[f"trace_clean_d{d}_s{s}"] = tr_cl
                arrays[f"trace_mix_d{d}_s{s}"]   = tr_mx
        dr = [r for r in rows if r["d"] == d]
        mr = np.mean([r["residual"] for r in dr])
        mc = np.mean([r["pct_clean"] for r in dr])
        mo = np.mean([r["occ_mix"] for r in dr])
        mdm = np.mean([r["pct_deployed_mix"] for r in dr])
        mdo = np.mean([r["occ_deployed_mix"] for r in dr])
        print(f"  d={d:3d}  resid={mr:+.3f}  clean={100*mc:4.0f}%  actor_occ={mo:.3f}"
              f"  deployed_mix={100*mdm:4.0f}% occ={mdo:.3f}   [{time.time()-t_start:5.0f}s]")

    config = dict(
        experiment="expB_capacity", size=SIZE, start=START, goal=GOAL,
        distractor=DCELL, r_int=R_INT, distractor_idx=didx,
        beta=BETA, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI, tau=TAU,
        gamma=panel.gamma, horizon=env.max_steps, J_opt=J_opt,
        d_grid=list(D_GRID), seeds=list(SEEDS), n_iters=N_ITERS,
    )
    save_run(OUTDIR, "expB_capacity", config, rows,
             arrays=(arrays if SAVE_TRACES else None))

    print("\n=== pooled residual curve (mean +/- std over seeds) ===")
    print(f"{'d':>4}  {'residual':>16}  {'clean%':>7}  {'mixact%':>7}  {'actor_occ':>9}  "
          f"{'deploy%':>7}  {'deploy_occ':>10}")
    headline = None
    for d in D_GRID:
        dr = [r for r in rows if r["d"] == d]
        res = np.array([r["residual"] for r in dr])
        print(f"{d:>4}  {res.mean():>+8.3f} +/-{res.std():>5.3f}  "
              f"{100*np.mean([r['pct_clean'] for r in dr]):>6.0f}%  "
              f"{100*np.mean([r['pct_mix'] for r in dr]):>6.0f}%  "
              f"{np.mean([r['occ_mix'] for r in dr]):>9.3f}  "
              f"{100*np.mean([r['pct_deployed_mix'] for r in dr]):>6.0f}%  "
              f"{np.mean([r['occ_deployed_mix'] for r in dr]):>10.3f}")
        if headline is None and res.mean() <= TAU_RESID:
            headline = d
    print(f"\nsuggested d (smallest with pooled residual <= {TAU_RESID}): {headline}")
    print("inspect the full curve / traces.")


if __name__ == "__main__":
    main()