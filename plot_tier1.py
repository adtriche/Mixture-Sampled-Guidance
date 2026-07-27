from __future__ import annotations
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE = "render"
FIGS_DIR = os.environ.get("SEC6_FIGS_DIR", os.path.join(os.getcwd(), "figs"))
EMIT_CAPACITY_TABLE = True
DPI = 300

ISO_ARM, ISO_BETA = "disjoint", 0.45
RB_LAM = 1.0

COLOR = {
    "actor":   "#1f77b4",
    "advisor": "#ff7f0e",
    "mix":     "#2ca02c",
    "rb":      "#d62728",
    "solo":    "#1f77b4",
}

from verify_section6 import _load, _arr, _by_seed, reduce_traj, JOPT

FRAC_NEAR0_FIELDS = ("adv_frac_near0",)

_FINAL_RE = re.compile(
    r"^final_(?P<pname>pi[A-Za-z]+)_(?P<arm>[a-z]+)_(?P<knob>[a-z]+)(?P<val>[0-9.]+)_s(?P<seed>\d+)$")


def _field(row, candidates):
    for c in candidates:
        if c in row:
            return float(row[c])
    raise KeyError(f"none of {candidates} present; row has {sorted(row.keys())[:8]}...")


def _seed_band(bs, key, x_key="iter"):
    grids = [np.array([float(r[x_key]) for r in s]) for s in bs.values()]
    x = grids[0]
    ys = np.vstack([[float(r[key]) for r in s] for s in bs.values()])
    return x, ys.mean(axis=0), ys.std(axis=0)


def _save(fig, name):
    os.makedirs(FIGS_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figs/{name}.png/.pdf")


Y_LABEL_WORDS = True
_YLAB = "fraction of optimal return" if Y_LABEL_WORDS else r"$J^{E}/J^{E}_{\mathrm{opt}}$"


def fig_control_combined():
    rows_s = _load("expC_control_shaped")
    rows_b = _load("expC_control_boltzmann")
    if not rows_s or not rows_b:
        print("  [skip] control_combined: missing one of the control arms")
        return
    bs_s, bs_b = _by_seed(rows_s), _by_seed(rows_b)
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(5.2, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    for key, label, ls in (("J_ext_actor", "actor (solo eval)", "-"),
                           ("J_ext_mix", "behavioral mixture", "-"),
                           ("J_ext_solo", "solo reference", "--")):
        col = COLOR["mix" if "mix" in key else ("solo" if "solo" in key else "actor")]
        x, m, s = _seed_band(bs_s, key)
        ax.plot(x, m / JOPT, ls, color=col, lw=1.6, label=label)
        ax.fill_between(x, np.clip((m - s) / JOPT, 0, None), (m + s) / JOPT,
                        color=col, alpha=0.18, lw=0)
    ax.set_ylabel(_YLAB)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    for bs, label, ls in ((bs_s, "shaped advisor", "-"),
                          (bs_b, "Boltzmann reference", ":")):
        x, m, s = _seed_band(bs, "beta")
        axb.plot(x, m, ls, color="#555555", lw=1.6, label=label)
        axb.fill_between(x, np.clip(m - s, 0, None), m + s,
                         color="#555555", alpha=0.14, lw=0)
    axb.set_ylabel(r"$\beta$")
    axb.set_xlabel("iteration")
    axb.set_ylim(-0.02, None)
    axb.legend(frameon=False, fontsize=8)
    _save(fig, "control_combined")


def fig_control(mode):
    rows = _load(f"expC_control_{mode}")
    if not rows:
        print(f"  [skip] expC_control_{mode}: no rows resolved")
        return
    bs = _by_seed(rows)
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(5.2, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    for key, label, ls in (("J_ext_actor", "actor (solo eval)", "-"),
                           ("J_ext_mix", "behavioral mixture", "-"),
                           ("J_ext_solo", "solo reference", "--")):
        col = COLOR["mix" if "mix" in key else ("solo" if "solo" in key else "actor")]
        x, m, s = _seed_band(bs, key)
        ax.plot(x, m / JOPT, ls, color=col, lw=1.6, label=label)
        ax.fill_between(x, (m - s) / JOPT, (m + s) / JOPT, color=col, alpha=0.18, lw=0)
    ax.set_ylabel(_YLAB)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    x, m, s = _seed_band(bs, "beta")
    axb.plot(x, m, color="#555555", lw=1.6)
    axb.fill_between(x, m - s, m + s, color="#555555", alpha=0.18, lw=0)
    axb.set_ylabel(r"$\beta$")
    axb.set_xlabel("iteration")
    axb.set_ylim(-0.02, None)
    _save(fig, f"control_{mode}")


def fig_control_supp(mode):
    rows = _load(f"expC_control_{mode}")
    if not rows:
        return
    bs = _by_seed(rows)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.0), sharex=True)
    x, m, s = _seed_band(bs, "I_e")
    ax1.plot(x, m, color="#555555", lw=1.6)
    ax1.fill_between(x, m - s, m + s, color="#555555", alpha=0.18, lw=0)
    ax1.set_ylabel(r"$I_e$")

    x, m, s = _seed_band(bs, "H_lambda")
    ax2.plot(x, m, color="#8c564b", lw=1.6)
    ax2.fill_between(x, m - s, m + s, color="#8c564b", alpha=0.18, lw=0)
    ax2.axhline(0.0, color="k", lw=0.6)
    if mode == "shaped":
        cross = []
        for s_rows in bs.values():
            for r in s_rows:
                if float(r["iter"]) >= 150 and float(r["H_lambda"]) < 0:
                    cross.append(float(r["iter"]))
                    break
        if cross:
            ax2.axvline(float(np.mean(cross)), color="#8c564b", lw=0.9, ls=":",
                        label=f"onset crossing (mean iter {np.mean(cross):.0f})")
            ax2.legend(frameon=False, fontsize=8)
    ax2.set_ylabel(r"$\bar H_\lambda$")
    ax2.set_xlabel("iteration")
    _save(fig, f"control_supp_{mode}")


_TAG_RE = re.compile(r"^trace_(?P<arm>[a-z]+)_(?P<knob>[a-z]+)(?P<val>[0-9.]+)_s(?P<seed>\d+)$")


def _expA_traces():
    z = _arr("expA_credit")
    if z is None:
        return None, None
    meta_cols = None
    try:
        from exp_credit import _TRACE_COLUMNS as meta_cols
    except Exception:
        pass
    out = {}
    for k in z.files:
        m = _TAG_RE.match(k)
        if m:
            out[k] = (m.groupdict(), z[k])
    return out, (list(meta_cols) if meta_cols else None)


def fig_isolation_curves():
    traces, cols = _expA_traces()
    if not traces:
        print("  [skip] expA arrays: no trace_* keys resolved")
        return
    if cols is None:
        print("  [skip] isolation curves: trace column order unavailable "
              "(exp_credit import failed); probe output will show it")
        return
    ci = {c: i for i, c in enumerate(cols)}

    def cell(arm, val):
        picks = [(g, a) for g, a in traces.values()
                 if g["arm"] == arm and abs(float(g["val"]) - val) < 1e-9]
        return picks

    fig, (axo, axj) = plt.subplots(2, 1, figsize=(5.2, 4.4), sharex=True)
    series = [("actor", "distr_occ_actor", "J_ext_actor", cell(ISO_ARM, ISO_BETA)),
              ("advisor", "distr_occ_advisor", "J_ext_advisor", cell(ISO_ARM, ISO_BETA)),
              ("mix", "distr_occ_mix", "J_ext_mix", cell(ISO_ARM, ISO_BETA)),
              ("rb", "distr_occ_rb", "J_ext_rb", cell("rb", RB_LAM))]
    for name, occ_key, j_key, picks in series:
        if not picks or occ_key not in ci:
            continue
        arrs = np.stack([a for _, a in picks])
        x = arrs[0][:, ci["iter"]]
        for axis, key in ((axo, occ_key), (axj, j_key)):
            ys = arrs[:, :, ci[key]]
            m, s = np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)
            if key.startswith("J_"):
                m, s = m / JOPT, s / JOPT
            lo, hi = m - s, m + s
            if key.startswith("J_") or key.startswith("distr_occ"):
                lo = np.clip(lo, 0.0, None)
            axis.plot(x, m, color=COLOR[name], lw=1.6, label=name)
            axis.fill_between(x, lo, hi, color=COLOR[name], alpha=0.18, lw=0)
    axo.set_ylabel("distractor occupancy")
    axo.legend(frameon=False, fontsize=8, ncol=2)
    axj.set_ylabel(_YLAB)
    axj.set_xlabel("iteration")
    _save(fig, "isolation_curves")


def fig_variance_ladder():
    b0s = ("020", "030", "040", "050", "060", "070", "080")
    styles = {"shaped": (COLOR["advisor"], "o", "co-trained shaped advisor"),
              "boltzmann": ("#7f7f7f", "s", "Boltzmann reference")}
    fig, (axr, axf) = plt.subplots(2, 1, figsize=(5.2, 4.4), sharex=True)
    for mode, (col, mk, label) in styles.items():
        xs, rm, rs, fmax, fpts = [], [], [], [], []
        for t in b0s:
            rows = _load(f"expD_sweep_b{t}_{mode}")
            if not rows:
                continue
            bs = _by_seed(rows)
            fin = [float(s[-1]["rd_ct2_over_mt2"]) for s in bs.values()]
            pks = [reduce_traj([_field(r, FRAC_NEAR0_FIELDS) for r in s], "peak")
                   for s in bs.values()]
            x = int(t) / 100.0
            xs.append(x); rm.append(np.mean(fin)); rs.append(np.std(fin))
            fmax.append(np.max(pks)); fpts.append((x, pks))
        axr.errorbar(xs, rm, yerr=rs, color=col, marker=mk, ms=4, lw=1.6,
                     capsize=2, label=label)
        axf.plot(xs, fmax, color=col, marker=mk, ms=4, lw=1.6)
        for x, pks in fpts:
            axf.plot([x] * len(pks), pks, ".", color=col, ms=3, alpha=0.35)
    axr.axhline(1.0, color="k", lw=0.6, ls=":")
    axr.set_ylabel("ct$_2$/mt$_2$ (final)")
    axr.legend(frameon=False, fontsize=8, loc="upper left")
    axf.set_ylabel("advisor mass near zero\n(peak per seed)")
    axf.set_xlabel(r"starting weight $\beta_0$")
    _save(fig, "variance_ladder")


def fig_isolation_heatmaps():
    z = _arr("expA_credit")
    if z is None:
        print("  [skip] heatmaps: expA arrays not resolved")
        return
    try:
        import exp_credit as XA
        from gridworld_env import GridEnv
        from panel import Panel
    except Exception as e:
        print(f"  [skip] heatmaps: repo imports failed ({e}); run from the repo root")
        return
    env = GridEnv(size=XA.SIZE, start=XA.START, goal=XA.GOAL,
                  distractors=[(XA.DCELL, XA.R_INT)])
    panel = Panel.from_env(env)
    per_seed = {}
    for k in z.files:
        m = _FINAL_RE.match(k)
        if not m:
            continue
        g = m.groupdict()
        if g["arm"] == ISO_ARM and abs(float(g["val"]) - ISO_BETA) < 1e-9:
            per_seed.setdefault(int(g["seed"]), {})[g["pname"]] = np.asarray(z[k])
    if not per_seed:
        print("  [skip] heatmaps: no finals matched the operating cell")
        return

    panels = {"actor": [], "advisor": [], "behavioral mixture": []}
    for s, pis in sorted(per_seed.items()):
        if "piA" not in pis or "piB" not in pis:
            continue
        piA, piB = pis["piA"], pis["piB"]
        piPhi = (1.0 - ISO_BETA) * piA + ISO_BETA * piB
        for label, pi in (("actor", piA), ("advisor", piB), ("behavioral mixture", piPhi)):
            panels[label].append(panel.solve(pi).d.reshape(env.size, env.size))

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    from matplotlib.colors import PowerNorm
    vmax = max(np.mean(v, axis=0).max() for v in panels.values() if v)
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)
    im = None
    for ax, (label, maps) in zip(axes, panels.items()):
        if not maps:
            ax.set_axis_off()
            continue
        im = ax.imshow(np.mean(maps, axis=0), cmap="viridis", norm=norm)
        gr, gc = XA.GOAL
        dr, dc = XA.DCELL
        sr, sc = XA.START
        ax.text(gc, gr, "G", color="white", ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.text(dc, dr, "D", color="white", ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.text(sc, sr, r"$S_0$", color="white", ha="center", va="center",
                fontsize=7, fontweight="bold")
        ax.set_title(f"{label} (n={len(maps)})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    _save(fig, "isolation_heatmaps")

def emit_variance_table():
    print("\n% ---- variance ladder (final ct2/mt2 by starting weight; paste into draft) ----")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    b0s = ("020", "030", "040", "050", "060", "070", "080")
    header = " & ".join(f"$\\beta_0={int(t)/100:.1f}$" for t in b0s[1:])
    rows_out = {"shaped": [], "boltzmann": []}
    frac_out = {"shaped": [], "boltzmann": []}
    for mode in ("shaped", "boltzmann"):
        for t in b0s[1:]:
            rows = _load(f"expD_sweep_b{t}_{mode}")
            if not rows:
                rows_out[mode].append("--"); frac_out[mode].append("--")
                continue
            bs = _by_seed(rows)
            fin = np.nanmean([float(s[-1]["rd_ct2_over_mt2"]) for s in bs.values()])
            pk = np.nanmax([reduce_traj([_field(r, FRAC_NEAR0_FIELDS) for r in s], "peak")
                            for s in bs.values()])
            rows_out[mode].append(f"{fin:.2f}")
            frac_out[mode].append(f"{pk:.2f}")
    print(f"advisor & {header} \\\\ \\midrule")
    print("shaped, ct$_2$/mt$_2$ & " + " & ".join(rows_out["shaped"]) + r" \\")
    print("shaped, frac$_{\\approx 0}$ (peak) & " + " & ".join(frac_out["shaped"]) + r" \\")
    print("Boltzmann, ct$_2$/mt$_2$ & " + " & ".join(rows_out["boltzmann"]) + r" \\")
    print("Boltzmann, frac$_{\\approx 0}$ (peak) & " + " & ".join(frac_out["boltzmann"]) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

def emit_capacity_table():
    rows = _load("expB_capacity")
    if not rows:
        return
    print("\n% ---- capacity residual by d (optional) ----")
    ds = sorted({int(float(r["d"])) for r in rows})
    print(r"\begin{tabular}{l" + "c" * len(ds) + "}")
    print(r"\toprule")
    print("d & " + " & ".join(str(d) for d in ds) + r" \\ \midrule")
    for red, label in (("mean", "residual (mean)"), ("peak", "residual (max)")):
        vals = []
        for d in ds:
            col = np.array([float(r["residual"]) for r in rows if int(float(r["d"])) == d])
            vals.append(f"{(col.mean() if red == 'mean' else col.max()):.3f}")
        print(f"{label} & " + " & ".join(vals) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def probe():
    print("== PROBE: paste this output back before rendering ==\n")
    z = _arr("expA_credit")
    if z is None:
        print("expA_credit arrays: NOT RESOLVED (check SEC6_RUNS_DIR)")
    else:
        keys = sorted(z.files)
        tr = [k for k in keys if k.startswith("trace_")]
        fin = [k for k in keys if k.startswith("final_")]
        print(f"expA npz: {len(tr)} trace keys, {len(fin)} final keys")
        print("  trace key examples :", tr[:3])
        print("  final key examples :", fin[:6])
    try:
        from exp_credit import _TRACE_COLUMNS
        print("  trace columns      :", list(_TRACE_COLUMNS))
    except Exception as e:
        print("  trace columns      : exp_credit import failed:", e)
    rows = _load("expD_sweep_b080_shaped")
    if rows:
        print("expD b080 shaped row fields:", sorted(rows[0].keys()))
    for modname in ("occ_view", "panel"):
        try:
            mod = __import__(modname)
            pub = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
            print(f"{modname} public callables:", pub)
        except Exception as e:
            print(f"{modname}: import failed: {e}")


def main():
    if MODE == "probe":
        probe()
        return
    print("rendering Tier 1 figures ->", FIGS_DIR)
    fig_control_combined()
    for mode in ("shaped", "boltzmann"):
        fig_control(mode)
        fig_control_supp(mode)
    fig_isolation_curves()
    fig_isolation_heatmaps()
    fig_variance_ladder()
    emit_variance_table()
    if EMIT_CAPACITY_TABLE:
        emit_capacity_table()


if __name__ == "__main__":
    main()