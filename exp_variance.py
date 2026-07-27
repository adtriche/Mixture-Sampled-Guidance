import os
import csv
import time
import numpy as np
from apparatus import Apparatus, FeatureMap, boltzmann_advisor
from recording import save_run
from exp_control import (
    build, find_lambda_ceiling, fd_dial_check,
    TAU, LR, DELTA_C, DELTA_PHI, D_FEAT, J_OPT,
    DBUDGET, LAM_WITHDRAW_BETA, DLAM, GBETA_TRIGGER_PATIENCE,
)

BETA0 = 0.80
BETA0_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ADVISOR_MODES = ("boltzmann", "shaped")

SEEDS = list(range(8))
N_ITERS = 2500
TRACE_EVERY = 25
FD_CHECKPOINTS = [0, 100, 250, 500, 1000, 1500, 2000, N_ITERS - 1]
SNAP_CHECKPOINTS = FD_CHECKPOINTS
LAM_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
OUTDIR = "runs/expD"
RUN_NAME = "expD_variance"
SWEEP_NAME = "expD_sweep"

SUMMARY_COLS = [
    "beta", "competence", "D_e", "J_ext_advisor", "J_ext_solo",
    "rd_ct2_over_mt2", "rd_max_IS", "rd_mt2_over_bound",
    "adv_ct2_over_mt2", "adv_max_IS", "adv_mt2_over_bound",
    "rd_shaped_ct2_over_mt2", "rd_shaped_mt2_over_bound",
    "adv_shaped_ct2_over_mt2", "adv_shaped_mt2_over_bound",
    "adv_entropy_occ", "adv_max_piB", "adv_frac_near0",
]

def _ratio_scalars(rr):
    rA, rphi, rAphi, wA = (np.asarray(rr.r_A), np.asarray(rr.r_phi), np.asarray(rr.r_Aphi), np.asarray(rr.w_A))
    IS = wA / (1.0 - rr.beta)
    j = int(np.argmax(rAphi))
    mt2, ct2, bnd = rr.mt_second_moment, rr.ct_second_moment, rr.mt_uniform_sm_bound
    return {
        "rd_A_max": rr.A_max,
        "rd_mt_mean": rr.mt_mean, "rd_mt_sm": mt2, "rd_mt_var": rr.mt_var,
        "rd_ct_mean": rr.ct_mean, "rd_ct_sm": ct2, "rd_ct_var": rr.ct_var,
        "rd_mt_bound": bnd,
        "rd_ct2_over_mt2": (ct2 / mt2) if mt2 > 0 else np.nan,
        "rd_mt2_over_bound": (mt2 / bnd) if bnd else np.nan,
        "rd_max_r_Aphi": float(rAphi.max()), "rd_max_r_phi": float(rphi.max()),
        "rd_max_r_A": float(rA.max()), "rd_max_w_A": float(wA.max()),
        "rd_max_IS": float(IS.max()),
        "rd_peak_r_Aphi": float(rAphi.flat[j]),
        "rd_peak_r_A": float(rA.flat[j]),
        "rd_peak_IS": float(IS.flat[j]),
        "rd_peak_state": int(j // rA.shape[1]), "rd_peak_action": int(j % rA.shape[1]),
    }


def _recovery_scalars(rc):
    return {
        "rec_L_phi": rc.L_phi, "rec_L_bar": rc.L_bar, "rec_L_A": rc.L_A,
        "rec_occ_channel": rc.occ_channel, "rec_occ_bound": rc.occ_bound,
        "rec_int_channel": rc.int_channel, "rec_int_bound": rc.int_bound,
        "rec_total": rc.total, "rec_total_bound": rc.total_bound,
        "rec_I_e": rc.I_e, "rec_A_max": rc.A_max,
    }


_ADV_EXT_KEYS = ("adv_mt_sm", "adv_ct_sm", "adv_ct2_over_mt2", "adv_max_IS",
                 "adv_mt2_over_bound", "adv_r_B_max")
_ADV_SH_KEYS = ("adv_shaped_mt_sm", "adv_shaped_ct_sm", "adv_shaped_ct2_over_mt2",
                "adv_shaped_A_max", "adv_shaped_mt2_over_bound")


def _moments_under(r_phi, r_Aphi, rho, A, delta_phi):
    A = np.asarray(A, dtype=np.float64); rho = np.asarray(rho)
    A_max = float(np.max(np.abs(A)))
    Xp = np.asarray(r_phi) * A; Xc = np.asarray(r_Aphi) * A
    mt2 = float((rho * Xp * Xp).sum()); ct2 = float((rho * Xc * Xc).sum())
    bnd = (1.0 + delta_phi) ** 2 * A_max ** 2
    return {"mt_sm": mt2, "ct_sm": ct2,
            "ct2_over_mt2": (ct2 / mt2) if mt2 > 0 else np.nan,
            "A_max": A_max, "mt2_over_bound": (mt2 / bnd) if bnd > 0 else np.nan}


def _advisor_sharpness(pi_B, d_state):
    pB = np.asarray(pi_B, dtype=np.float64)
    ent = -(pB * np.log(np.clip(pB, 1e-12, None))).sum(axis=1)
    return {
        "adv_entropy_occ": float(np.asarray(d_state) @ ent),
        "adv_max_piB": float(pB.max()),
        "adv_frac_near0": float((pB < 0.01).mean()),
    }


def _readout(panel, pi_A, pi_B, beta, pi_A_plus, pi_B_plus, lam, want_arrays):
    sol = panel.solve(panel.mixture(pi_A, pi_B, beta))
    rho = np.asarray(sol.rho)
    A_ext = np.asarray(sol.A_ext)
    A_sh = A_ext + lam * np.asarray(sol.A_int)
    rr = panel.ratio_diagnostics(pi_A, pi_B, beta, pi_A_plus, delta_phi=DELTA_PHI)
    rc = panel.surrogate_recovery(pi_A, pi_B, beta, pi_A_plus, delta_c=DELTA_C, delta_phi=DELTA_PHI)
    scal = {}
    scal.update(_ratio_scalars(rr))
    m = _moments_under(rr.r_phi, rr.r_Aphi, rho, A_sh, DELTA_PHI)
    scal.update({f"rd_shaped_{k}": v for k, v in m.items()})
    scal.update(_recovery_scalars(rc))
    scal.update(_advisor_sharpness(pi_B, np.asarray(sol.d)))

    if pi_B_plus is not None:
        ra = panel.ratio_diagnostics(pi_B, pi_A, 1.0 - beta, pi_B_plus, delta_phi=DELTA_PHI)
        IS = np.asarray(ra.w_A) / beta
        mt2, ct2, bnd = ra.mt_second_moment, ra.ct_second_moment, ra.mt_uniform_sm_bound
        scal.update({
            "adv_mt_sm": mt2, "adv_ct_sm": ct2,
            "adv_ct2_over_mt2": (ct2 / mt2) if mt2 > 0 else np.nan,
            "adv_max_IS": float(IS.max()),
            "adv_mt2_over_bound": (mt2 / bnd) if bnd else np.nan,
            "adv_r_B_max": float(np.asarray(ra.r_A).max()),
        })
        m2 = _moments_under(ra.r_phi, ra.r_Aphi, rho, A_sh, DELTA_PHI)
        scal.update({f"adv_shaped_{k}": v for k, v in m2.items()})
    else:
        scal.update({k: np.nan for k in _ADV_EXT_KEYS})
        scal.update({k: np.nan for k in _ADV_SH_KEYS})

    arrays = None
    if want_arrays:
        arrays = dict(
            piA=np.asarray(pi_A), piB=np.asarray(pi_B), piA_plus=np.asarray(pi_A_plus),
            r_A=np.asarray(rr.r_A), r_phi=np.asarray(rr.r_phi), r_Aphi=np.asarray(rr.r_Aphi),
            w_A=np.asarray(rr.w_A), w_B=np.asarray(rr.w_B),
            A_ext=A_ext, A_int=np.asarray(sol.A_int),
            rho_phi=rho, d_phi=np.asarray(sol.d),
        )
        if pi_B_plus is not None:
            arrays["piB_plus"] = np.asarray(pi_B_plus)
    return scal, arrays


def run(n_iters=N_ITERS, seed=0, advisor_mode="boltzmann", beta0=BETA0, gamma=None, verbose=True):
    if advisor_mode not in ("boltzmann", "shaped"):
        raise ValueError(f"advisor_mode must be 'boltzmann' or 'shaped', got {advisor_mode!r}")
    env, panel, room_states = build()
    g = env.gamma if gamma is None else gamma
    S, A = env.n_states, env.n_actions
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    pi_A0 = np.full((S, A), 1.0 / A)

    lam_star, Hgrid = find_lambda_ceiling(panel, pi_A0, beta0, TAU, LAM_GRID)
    if verbose:
        print(f"[find] lambda* (Boltzmann ceiling at initial actor, beta0={beta0}) "
              f"= {lam_star:.3f}   advisor={advisor_mode}")

    ap = Apparatus(panel, fm, A, advisor_mode=advisor_mode, beta=beta0, lam=lam_star,
                   tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                   distractor_states=room_states, seed=seed)
    solo = Apparatus(panel, fm, A, advisor_mode="none", beta=0.0,
                     tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI, seed=seed)

    checkpoints = sorted(set([c for c in SNAP_CHECKPOINTS if c < n_iters] + [n_iters - 1]))
    beta, lam = beta0, lam_star
    withdrawing = False
    g_pos_streak = 0
    traj, fd_rows, snaps = [], [], {}

    for e in range(n_iters):
        traced = (e % TRACE_EVERY == 0) or (e == n_iters - 1) or (e in checkpoints)
        pi_A_pre = ap.polA.probs().copy() if traced else None
        pi_B_roll = ap._advisor_probs(pi_A_pre, beta, lam) if traced else None

        rec = ap.step(beta=beta, lam=lam)
        solo.step()
        G, D = rec["G_beta"], rec["D_e"]

        if not withdrawing:
            g_pos_streak = g_pos_streak + 1 if G > 0 else 0
            if g_pos_streak >= GBETA_TRIGGER_PATIENCE:
                withdrawing = True
        if withdrawing:
            dbeta = (1.0 - g) / g * DBUDGET / max(D, 1e-6)
            beta = max(0.0, beta - dbeta)
            if beta < LAM_WITHDRAW_BETA:
                lam = max(0.0, lam - DLAM)

        if traced:
            pi_A_plus = ap.polA.probs().copy()
            pi_B_plus = ap.polB.probs().copy() if ap.polB is not None else None
            want_arrays = (e in checkpoints)
            scal, arrays = _readout(panel, pi_A_pre, pi_B_roll, rec["beta"],
                                    pi_A_plus, pi_B_plus, rec["lam"], want_arrays)
            row = dict(
                iter=e, beta=rec["beta"], lam=rec["lam"],
                withdrawing=int(withdrawing), g_pos_streak=g_pos_streak,
                J_ext_actor=rec["J_ext_actor"], J_ext_mix=rec["J_ext_mix"],
                J_ext_advisor=rec["J_ext_advisor"], J_ext_solo=solo.log[-1]["J_ext_actor"],
                competence=rec["J_ext_actor"] / J_OPT,
                G_beta=G, dJ_dbeta=rec["dJ_dbeta"],
                H_lambda=rec["H_lambda"], dJ_dlambda=rec["dJ_dlambda"],
                D_e=D, I_e=rec["I_e"], room_occ=rec["distr_occ_mix"],
            )
            row.update(scal)
            traj.append(row)
            if want_arrays:
                for k, v in arrays.items():
                    snaps[f"snap_{k}_s{seed}_it{e}"] = v

        if e in checkpoints:
            bp, br, lp, lr_ = fd_dial_check(panel, ap, beta, lam, TAU)
            fd_rows.append((e, bp, br, lp, lr_))

    finals = {"piA": ap.polA.probs(),
              "piB": ap._advisor_probs(ap.polA.probs(), beta, lam),
              "piA_solo": solo.polA.probs()}
    return dict(lam_star=lam_star, Hgrid=Hgrid, traj=traj, fd=fd_rows, snaps=snaps,
                finals=finals, beta0=beta0, advisor_mode=advisor_mode, gamma=g)


def _aggregate_and_save(per_seed, seeds, advisor_mode, n_iters, beta0, outdir, name):
    n = len(seeds)
    n_tr = len(per_seed[0]["traj"])
    arrays = {}
    for k in per_seed[0]["traj"][0].keys():
        try:
            arrays[k] = np.array([[per_seed[si]["traj"][i].get(k, np.nan)
                                   for i in range(n_tr)] for si in range(n)], dtype=float)
        except (TypeError, ValueError):
            pass
    arrays["seeds"] = np.array(seeds)
    arrays["lam_star"] = np.array([r["lam_star"] for r in per_seed])
    arrays["Hgrid"] = np.array([r["Hgrid"] for r in per_seed])
    arrays["lam_grid"] = np.array(LAM_GRID)

    fd_iters = np.array([row[0] for row in per_seed[0]["fd"]])
    def fdcol(j):
        return np.array([[r["fd"][k][j] for k in range(len(fd_iters))] for r in per_seed])
    arrays.update(fd_iters=fd_iters, fd_beta_pred=fdcol(1), fd_beta_real=fdcol(2),
                  fd_lam_pred=fdcol(3), fd_lam_real=fdcol(4))

    for r in per_seed:
        arrays.update(r["snaps"])
    for si, s in enumerate(seeds):
        for pname, parr in per_seed[si]["finals"].items():
            arrays[f"final_{pname}_s{s}"] = np.asarray(parr)

    log = []
    for si, s in enumerate(seeds):
        for row in per_seed[si]["traj"]:
            log.append(dict(seed=int(s), **row))

    config = dict(
        experiment="expD_variance", advisor_mode=advisor_mode, beta0=beta0,
        d_feat=D_FEAT, tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
        dbudget=DBUDGET, lam_withdraw_beta=LAM_WITHDRAW_BETA, dlam=DLAM,
        gbeta_patience=GBETA_TRIGGER_PATIENCE, lam_grid=list(LAM_GRID),
        seeds=list(seeds), n_iters=n_iters, trace_every=TRACE_EVERY,
        snap_checkpoints=[int(c) for c in fd_iters], j_opt=J_OPT,
        gamma=float(per_seed[0]["gamma"]),
        instrument="basin (two-room) from exp_control.build()",
        note=("controlled co-learning (controller on beta+lambda, mixture-targeted "
              "projected update, exact panel) from beta0; lambda* via Boltzmann ceiling "
              "applied to the deployed advisor. ACTOR side = ratio_diagnostics(piA,piB,"
              "beta,piA+); ADVISOR side = ratio_diagnostics(piB,piA,1-beta,piB+) on the "
              "SAME A^E (extrinsic footing, NOT the advisor's shaped objective) -- "
              "directly comparable; ceilings 1/(1-beta) vs 1/beta locate the hazard on "
              "the minority component. ct2 is a counterfactual on the realized update "
              "(other component held). Each side is recorded on TWO advantage footings: "
              "extrinsic A^E (rd_*/adv_*) and shaped A^E+lambda*A^I (rd_shaped_*/"
              "adv_shaped_*), reusing the advantage-independent ratios, so the bound "
              "mt2/bound<=1 is checked across {actor,advisor}x{extrinsic,shaped} -- the "
              "variance control is advantage-agnostic (covers the advisor's shaped "
              "update, not just the actor's credit-closed one). adv_* are NaN for "
              "boltzmann (no PG advisor)."),
        array_keys=("scalar columns = [n_seed, n_trace]; snap_<field>_s<seed>_it<iter> "
                    "= (S,A)/(S,) checkpoint arrays (incl. piB_plus where defined); "
                    "final_{piA,piB,piA_solo}_s<seed> = converged policies."),
    )
    paths = save_run(outdir, name, config, log, arrays=arrays)
    n_snap = sum(k.startswith("snap_") for k in arrays)
    return arrays, paths, len(log), n_snap


def report_pooled(A, mode="", beta0=None):
    it = A["iter"][0]
    ck = [c for c in (0, 250, 500, 1000, 1500, 2000) if c <= it[-1]] + [int(it[-1])]
    tag = f"{mode} beta0={beta0}" if beta0 is not None else mode
    print(f"\n=== pooled trajectory [{tag}] (mean+/-std over seeds) ===")
    print(" iter |   beta         comp        | act ct2/mt2   act maxIS  | adv ct2/mt2   adv maxIS  | mt2/bnd")
    for c in ck:
        i = int(np.argmin(np.abs(it - c)))
        def ms(name):
            if name not in A:
                return "    nan     "
            v = A[name][:, i]; v = v[np.isfinite(v)]
            return f"{v.mean():.3f}+/-{v.std():.3f}" if v.size else "    nan     "
        print(f"{int(it[i]):5d} | {ms('beta')} {ms('competence')} | "
              f"{ms('rd_ct2_over_mt2')} {ms('rd_max_IS')} | "
              f"{ms('adv_ct2_over_mt2')} {ms('adv_max_IS')} | {ms('rd_mt2_over_bound')}")
    def mx(k):
        return np.nanmax(A[k]) if k in A else float("nan")
    print(f"[{tag}] act ct2/mt2 max={mx('rd_ct2_over_mt2'):.2f}  adv ct2/mt2 max={mx('adv_ct2_over_mt2'):.2f}  "
          f"act maxIS={mx('rd_max_IS'):.2f}  adv maxIS={mx('adv_max_IS'):.2f}  "
          f"mt2/bound max={mx('rd_mt2_over_bound'):.4f} (<=1)")


def _summary_rows(arrays, mode, beta0):
    it = arrays["iter"][0]
    rows = []
    for i in range(len(it)):
        row = {"advisor_mode": mode, "beta0": float(beta0), "iter": int(it[i])}
        for c in SUMMARY_COLS:
            if c in arrays:
                v = arrays[c][:, i]; v = v[np.isfinite(v)]
                row[f"{c}_mean"] = float(v.mean()) if v.size else float("nan")
                row[f"{c}_std"] = float(v.std()) if v.size else float("nan")
            else:
                row[f"{c}_mean"] = float("nan"); row[f"{c}_std"] = float("nan")
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


def run_all(advisor_modes=ADVISOR_MODES, seeds=None, n_iters=None, beta0=None, outdir=None, name=None):
    seeds = SEEDS if seeds is None else list(seeds)
    n_iters = N_ITERS if n_iters is None else int(n_iters)
    beta0 = BETA0 if beta0 is None else float(beta0)
    outdir = OUTDIR if outdir is None else outdir
    name = RUN_NAME if name is None else name
    t0 = time.time()
    out = {}
    for mode in advisor_modes:
        print(f"\n===== Exp D | advisor={mode} | beta0={beta0} | seeds={seeds} =====")
        per_seed = []
        for s in seeds:
            res = run(n_iters=n_iters, seed=s, advisor_mode=mode, beta0=beta0, verbose=False)
            per_seed.append(res)
            end = res["traj"][-1]
            print(f"[{mode} seed {s}] lam*={res['lam_star']:.3f} final beta={end['beta']:.3f} "
                  f"actor={end['competence']:.3f} act ct2/mt2={end['rd_ct2_over_mt2']:.2f} "
                  f"adv ct2/mt2={end['adv_ct2_over_mt2']:.2f} [{time.time()-t0:.0f}s]")
        arrays, paths, n_rows, n_snap = _aggregate_and_save(
            per_seed, seeds, mode, n_iters, beta0, outdir, f"{name}_{mode}")
        report_pooled(arrays, mode, beta0)
        print(f"wrote {n_rows} rows, {n_snap} arrays [{mode}] -> {paths['log']}")
        out[mode] = (arrays, paths)
    print(f"\n[done in {time.time()-t0:.0f}s]")
    return out


def sweep_beta0(beta0_grid=None, advisor_modes=ADVISOR_MODES, seeds=None, n_iters=None, outdir=None, name=None):
    beta0_grid = BETA0_GRID if beta0_grid is None else list(beta0_grid)
    seeds = SEEDS if seeds is None else list(seeds)
    n_iters = N_ITERS if n_iters is None else int(n_iters)
    outdir = OUTDIR if outdir is None else outdir
    name = SWEEP_NAME if name is None else name
    t0 = time.time()
    summary = []
    print(f"### beta0 sweep {beta0_grid} x {list(advisor_modes)} x {len(seeds)} seeds "
          f"x {n_iters} iters ###")
    for b0 in beta0_grid:
        for mode in advisor_modes:
            tag = f"{name}_b{int(round(b0 * 100)):03d}_{mode}"
            print(f"\n===== beta0={b0:.2f} | advisor={mode} =====")
            per_seed = []
            for s in seeds:
                res = run(n_iters=n_iters, seed=s, advisor_mode=mode, beta0=b0, verbose=False)
                per_seed.append(res)
                end = res["traj"][-1]
                print(f"[b0={b0:.2f} {mode} s{s}] lam*={res['lam_star']:.3f} "
                      f"beta_f={end['beta']:.3f} actor={end['competence']:.3f} "
                      f"act={end['rd_ct2_over_mt2']:.2f} adv={end['adv_ct2_over_mt2']:.2f} "
                      f"[{time.time()-t0:.0f}s]")
            arrays, paths, n_rows, n_snap = _aggregate_and_save(
                per_seed, seeds, mode, n_iters, b0, outdir, tag)
            summary.extend(_summary_rows(arrays, mode, b0))
            report_pooled(arrays, mode, b0)
            print(f"wrote {n_rows} rows, {n_snap} arrays -> {tag}")
    spath = _write_summary(summary, outdir, f"{name}_summary")
    print(f"\n[sweep done in {time.time()-t0:.0f}s]  plot-ready summary: {spath}")
    return summary


if __name__ == "__main__":
    sweep_beta0()