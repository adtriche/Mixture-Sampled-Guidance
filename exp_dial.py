import os
import csv
import time
import numpy as np
from apparatus import Apparatus, FeatureMap, boltzmann_advisor
from recording import save_run
from exp_control import (build, find_lambda_ceiling, TAU, LR, DELTA_C, DELTA_PHI, D_FEAT, J_OPT,)

BETA_GRID = [0.45, 0.80]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
ADVISOR_MODES = ("boltzmann", "shaped")
SEEDS = list(range(8))
N_ITERS = 2500
TRACE_EVERY = 25
CONVERGE_FRAC = 0.20
OUTDIR = "runs/expC"
RUN_NAME = "expC_dial"

_FIELDS = ("iter", "beta", "lam", "J_ext_mix", "J_ext_advisor", "J_ext_actor",
           "J_ext_solo", "competence", "G_beta", "dJ_dbeta", "H_lambda",
           "dJ_dlambda", "D_e", "I_e", "room_occ")


def run(n_iters=N_ITERS, seed=0, advisor_mode="shaped", beta=0.45, lam=0.868, verbose=False):
    if advisor_mode not in ("boltzmann", "shaped"):
        raise ValueError(f"advisor_mode must be 'boltzmann' or 'shaped', got {advisor_mode!r}")
    env, panel, room_states = build()
    g = env.gamma
    S, A = env.n_states, env.n_actions
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    ap = Apparatus(panel, fm, A, advisor_mode=advisor_mode, beta=beta, lam=lam,
                   tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                   distractor_states=room_states, seed=seed)
    solo = Apparatus(panel, fm, A, advisor_mode="none", beta=0.0,
                     tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI, seed=seed)
    traj = []
    for e in range(n_iters):
        rec = ap.step(beta=beta, lam=lam)
        solo.step()
        if (e % TRACE_EVERY == 0) or (e == n_iters - 1):
            traj.append(dict(
                iter=e, beta=beta, lam=lam,
                J_ext_mix=rec["J_ext_mix"], J_ext_advisor=rec["J_ext_advisor"],
                J_ext_actor=rec["J_ext_actor"], J_ext_solo=solo.log[-1]["J_ext_actor"],
                competence=rec["J_ext_actor"] / J_OPT,
                G_beta=rec["G_beta"], dJ_dbeta=rec["dJ_dbeta"],
                H_lambda=rec["H_lambda"], dJ_dlambda=rec["dJ_dlambda"],
                D_e=rec["D_e"], I_e=rec["I_e"], room_occ=rec["distr_occ_mix"]))
    finals = {"piA": ap.polA.probs(),
              "piB": ap._advisor_probs(ap.polA.probs(), beta, lam),
              "piA_solo": solo.polA.probs()}
    return dict(traj=traj, finals=finals, beta=beta, lam=lam,
                advisor_mode=advisor_mode, gamma=g)


def _converged(traj):
    k = max(1, int(round(len(traj) * CONVERGE_FRAC)))
    tail = traj[-k:]
    return {f: float(np.mean([row[f] for row in tail]))
            for f in _FIELDS if f not in ("iter",)}


def _aggregate_and_save(per_seed, seeds, advisor_mode, beta, lam_grid, n_iters, outdir, name):
    n_lam, n_seed = len(lam_grid), len(seeds)
    n_tr = len(per_seed[0][0]["traj"])
    arrays = {}
    for f in _FIELDS:
        arrays[f] = np.array(
            [[[per_seed[li][si]["traj"][i][f] for i in range(n_tr)]
              for si in range(n_seed)] for li in range(n_lam)], dtype=float)
    arrays["lam_grid"] = np.array(lam_grid)
    arrays["seeds"] = np.array(seeds)
    for li, lv in enumerate(lam_grid):
        for si, s in enumerate(seeds):
            for pname, parr in per_seed[li][si]["finals"].items():
                arrays[f"final_{pname}_lam{int(round(lv*100)):03d}_s{s}"] = np.asarray(parr)

    log = []
    for li, lv in enumerate(lam_grid):
        for si, s in enumerate(seeds):
            for row in per_seed[li][si]["traj"]:
                log.append(dict(seed=int(s), lam_grid_value=float(lv), **row))

    config = dict(
        experiment="expC_dial", advisor_mode=advisor_mode, beta=beta,
        lam_grid=list(lam_grid), seeds=list(seeds), n_iters=n_iters,
        trace_every=TRACE_EVERY, converge_frac=CONVERGE_FRAC,
        d_feat=D_FEAT, tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
        j_opt=J_OPT, gamma=float(per_seed[0][0]["gamma"]),
        instrument="basin (two-room) from exp_control.build()",
        note=("fixed-control co-training (actor+advisor co-evolve, beta and lambda HELD) "
              "swept over lambda at a held beta. Realized J^E_mix(lambda) curve vs the "
              "logged leading-order H_lambda prediction (control_gradient on the trained "
              "rollout advisor) and its zero crossing. boltzmann = exact closed-form "
              "reference; shaped = learned-advisor leading-order test. Read at held beta "
              "where the advisor is trained. Dial-validation perturbation in lambda, NOT "
              "the A/B two-channel sweep nor the Exp C withdrawal handoff."),
        array_keys="scalar fields = [n_lam, n_seed, n_trace]; lam axis = lam_grid.",
    )
    paths = save_run(outdir, name, config, log, arrays=arrays)
    return arrays, paths, len(log)


def _summary_rows(per_seed, seeds, advisor_mode, beta, lam_grid):
    rows = []
    for li, lv in enumerate(lam_grid):
        conv = [_converged(per_seed[li][si]["traj"]) for si in range(len(seeds))]
        row = {"advisor_mode": advisor_mode, "beta": float(beta), "lam": float(lv)}
        for f in conv[0]:
            v = np.array([c[f] for c in conv], dtype=float)
            row[f"{f}_mean"] = float(np.nanmean(v))
            row[f"{f}_std"] = float(np.nanstd(v))
        rows.append(row)
    return rows


def _write_summary(summary, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    return path


def report(arrays, advisor_mode, beta):
    lam = arrays["lam_grid"]
    k = max(1, int(round(arrays["J_ext_mix"].shape[2] * CONVERGE_FRAC)))
    def conv(field):                       # [n_lam] seed+tail mean
        return np.nanmean(arrays[field][:, :, -k:], axis=(1, 2))
    Jmix, Jadv, comp = conv("J_ext_mix") / J_OPT, conv("J_ext_advisor") / J_OPT, conv("competence")
    Hl, dJl = conv("H_lambda"), conv("dJ_dlambda")
    lam_star_pred = _zero_crossing(lam, Hl)
    lam_star_real = lam[int(np.argmax(Jmix))]
    print(f"\n=== [{advisor_mode} beta={beta}] converged dial curve over lambda ===")
    print(f"  {'lam':>5} {'Jmix/Jopt':>10} {'Jadv/Jopt':>10} {'comp':>6} {'H_lambda':>10} {'dJ_dlam':>10}")
    for i in range(len(lam)):
        print(f"  {lam[i]:5.2f} {Jmix[i]:10.4f} {Jadv[i]:10.4f} {comp[i]:6.3f} {Hl[i]:10.4f} {dJl[i]:10.4f}")
    print(f"  over-shaping onset: realized peak lambda*={lam_star_real:.3f}  "
          f"vs H_lambda=0 crossing lambda*={lam_star_pred}")


def _zero_crossing(x, y):
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if y[i + 1] == y[i]:
        return float(x[i])
    return round(float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])), 3)


def sweep_lambda(beta_grid=None, lam_grid=None, advisor_modes=ADVISOR_MODES, seeds=None, n_iters=None, outdir=None, name=None):
    beta_grid = BETA_GRID if beta_grid is None else list(beta_grid)
    lam_grid = LAM_GRID if lam_grid is None else list(lam_grid)
    seeds = SEEDS if seeds is None else list(seeds)
    n_iters = N_ITERS if n_iters is None else int(n_iters)
    outdir = OUTDIR if outdir is None else outdir
    name = RUN_NAME if name is None else name
    t0 = time.time()
    summary = []
    print(f"### Exp C lambda-dial sweep | beta {beta_grid} x lambda {lam_grid} "
          f"x {list(advisor_modes)} x {len(seeds)} seeds x {n_iters} iters ###")
    for beta in beta_grid:
        for mode in advisor_modes:
            tag = f"{name}_b{int(round(beta * 100)):03d}_{mode}"
            print(f"\n===== beta={beta:.2f} | advisor={mode} =====")
            per_seed = []
            for lv in lam_grid:
                row = []
                for s in seeds:
                    res = run(n_iters=n_iters, seed=s, advisor_mode=mode,
                              beta=beta, lam=lv, verbose=False)
                    row.append(res)
                end = _converged(row[-1]["traj"])
                print(f"[b={beta:.2f} {mode} lam={lv:.2f}] Jmix/Jopt={end['J_ext_mix']/J_OPT:.4f} "
                      f"comp={end['competence']:.3f} H_lam={end['H_lambda']:.4f} "
                      f"dJ_dlam={end['dJ_dlambda']:.4f} [{time.time()-t0:.0f}s]")
                per_seed.append(row)
            arrays, paths, n_rows = _aggregate_and_save(
                per_seed, seeds, mode, beta, lam_grid, n_iters, outdir, tag)
            summary.extend(_summary_rows(per_seed, seeds, mode, beta, lam_grid))
            report(arrays, mode, beta)
            print(f"wrote {n_rows} rows -> {tag}")
    spath = _write_summary(summary, outdir, f"{name}_summary")
    print(f"\n[dial sweep done in {time.time()-t0:.0f}s]  plot-ready summary: {spath}")
    return summary


if __name__ == "__main__":
    sweep_lambda()