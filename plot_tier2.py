from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verify_tier2 import RUNS_T2, TAG_MAP, _families, _load_family  # noqa: E402

MODE = "render"
FIGS_DIR = os.environ.get("T2_FIGS_DIR", os.path.join(os.getcwd(), "figs"))
DPI = 300
SHOW_DEPLOYED_MSG = False

OCC = {
    "map_key": "deployed_occ_snapshots",
    "steps_key": "deployed_occ_snapshot_steps",
    "final_window": 10,
}

COLOR = {
    "msg":      "#2ca02c",
    "vanilla":  "#1f77b4",
    "rb":       "#d62728",
    "rbanneal": "#8c564b",
}
LS = {"msg": "-", "vanilla": "--", "rb": "-", "rbanneal": "-"}
LABEL = {
    "msg": "MSG (mixture)",
    "vanilla": "vanilla (VA)",
    "rb": "RB",
    "rbanneal": "anneal-RB",
}

ENVS = (
    ("DoorKey", {"msg": "doorkey_bb", "vanilla": "doorkey_vanilla",
                 "rb": "doorkey_rb", "rbanneal": "doorkey_rbanneal"}),
    ("FourRooms", {"msg": "fourrooms_bb", "vanilla": "fourrooms_vanilla",
                   "rb": "fourrooms_rb", "rbanneal": "fourrooms_rbanneal"}),
)


def _series(fam, field, ffill=False):
    steps = np.array([float(r["step"]) for r in next(iter(fam.values()))])
    rows = []
    for seed_rows in fam.values():
        v = np.array([float(r[field]) if r[field] != "" else np.nan for r in seed_rows])
        if ffill:
            last = np.nan
            for i in range(len(v)):
                if np.isnan(v[i]):
                    v[i] = last
                else:
                    last = v[i]
        rows.append(v)
    return steps, np.vstack(rows)


def _band(ax, x, mat, color, label, ls="-"):
    valid = ~np.all(np.isnan(mat), axis=0)
    x, mat = x[valid], mat[:, valid]
    m, s = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
    ax.plot(x, m, ls, color=color, lw=1.6, label=label)
    ax.fill_between(x, np.clip(m - s, 0.0, 1.0), np.clip(m + s, 0.0, 1.0),
                    color=color, alpha=0.15, lw=0)

def _save(fig, name):
    os.makedirs(FIGS_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figs/{name}.png/.pdf")


def fig_success_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, (env_name, fams) in zip(axes, ENVS):
        for arm in ("vanilla", "rbanneal", "rb", "msg"):
            fam = _load_family(fams[arm])
            if fam is None:
                print(f"  [skip] {env_name}/{arm}: unresolved")
                continue
            if arm == "msg":
                x, mat = _series(fam, "actor_eval_success", ffill=True)
                _band(ax, x / 1000.0, mat, COLOR[arm], LABEL[arm])
                if SHOW_DEPLOYED_MSG:
                    x2, mat2 = _series(fam, "mix_success")
                    _band(ax, x2 / 1000.0, mat2, COLOR[arm], "MSG (deployed)", ls="--")
            else:
                x, mat = _series(fam, "success_rate")
                _band(ax, x / 1000.0, mat, COLOR[arm], LABEL[arm])
        ax.set_title(env_name, fontsize=10)
        ax.set_xlabel("environment steps (thousands)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("success rate")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "t2_success_curves")


def fig_beta_trajectories():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.8), sharey=True)
    for ax, (env_name, fams) in zip(axes, ENVS):
        fam = _load_family(fams["msg"])
        if fam is None:
            continue
        x, mat = _series(fam, "beta")
        for row in mat:                                       # per-seed spaghetti
            ax.plot(x / 1000.0, row, color=COLOR["msg"], lw=0.6, alpha=0.30)
        ax.plot(x / 1000.0, np.nanmean(mat, axis=0), color=COLOR["msg"], lw=2.0)
        ax.set_title(env_name, fontsize=10)
        ax.set_xlabel("environment steps (thousands)")
        ax.set_ylim(-0.02, 0.5)
    axes[0].set_ylabel(r"$\beta$")
    _save(fig, "t2_beta_trajectories")

def _occ_paths(fam_key):
    import glob
    from verify_tier2 import TAG_MAP as TM, RUNS_T2 as RT
    tag = TM[fam_key]
    out = {}
    for arm in ("bb", "rb", "rbanneal", "vanilla"):
        for p in glob.glob(os.path.join(RT, arm, f"{tag}_seed*_occ.npz")):
            base = os.path.basename(p)
            s = int(base.split("_seed")[-1].split("_")[0])
            out[s] = p
    return out


def fig_t2_combined():
    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 4.6), sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    for c, (env_name, fams) in enumerate(ENVS):
        ax, axb = axes[0, c], axes[1, c]
        for arm in ("vanilla", "rbanneal", "rb", "msg"):
            fam = _load_family(fams[arm])
            if fam is None:
                print(f"  [skip] {env_name}/{arm}: unresolved")
                continue
            field = "mix_ext_return" if arm == "msg" else "ext_return"
            x, mat = _series(fam, field)
            _band(ax, x / 1000.0, mat, COLOR[arm], LABEL[arm], ls=LS[arm])
        ax.set_title(env_name, fontsize=10)
        ax.set_ylim(-0.02, 1.02)

        fam = _load_family(fams["msg"])
        if fam is not None:
            x, mat = _series(fam, "beta")
            for row in mat:
                axb.plot(x / 1000.0, row, color="#555555", lw=0.5, alpha=0.30)
            axb.plot(x / 1000.0, np.nanmean(mat, axis=0), color="#555555", lw=1.8)
        axb.set_xlabel("environment steps (thousands)")
        axb.set_ylim(-0.02, 0.5)
    axes[0, 0].set_ylabel("extrinsic return")
    axes[1, 0].set_ylabel(r"$\beta$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "t2_combined")


def fig_occupancy():
    from matplotlib.colors import PowerNorm
    order = ("msg", "vanilla", "rbanneal", "rb")
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.6))
    for r, (env_name, fams) in enumerate(ENVS):
        maps = {}
        for arm in order:
            paths = _occ_paths(fams[arm])
            per_seed = []
            for s, p in sorted(paths.items()):
                z = np.load(p, allow_pickle=True)
                stack = np.asarray(z[OCC["map_key"]], dtype=np.float64)
                per_seed.append(stack[-OCC["final_window"]:].mean(axis=0))
            if per_seed:
                maps[arm] = np.mean(per_seed, axis=0)
        if not maps:
            continue
        vmax = max(m.max() for m in maps.values())
        norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)
        im = None
        for c, arm in enumerate(order):
            ax = axes[r, c]
            if arm not in maps:
                ax.set_axis_off()
                continue
            im = ax.imshow(maps[arm], cmap="viridis", norm=norm)
            if r == 0:
                ax.set_title(LABEL[arm].replace(" (actor eval)", ""), fontsize=9)
            if c == 0:
                ax.set_ylabel(env_name, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        if im is not None:
            fig.colorbar(im, ax=list(axes[r]), shrink=0.9, pad=0.01)
    _save(fig, "t2_occupancy")


def probe():
    print("== TIER 2 PLOT PROBE: npz key inventory ==\n")
    import glob
    for key, tag in TAG_MAP.items():
        for arm in ("bb", "rb", "rbanneal", "vanilla"):
            hits = glob.glob(os.path.join(RUNS_T2, arm, f"{tag}_seed0_occ.npz"))
            if not hits:
                continue
            z = np.load(hits[0], allow_pickle=True)
            print(f"[{key}] {os.path.basename(hits[0])}")
            for k in z.files:
                arr = z[k]
                shp = getattr(arr, "shape", "?")
                print(f"    {k:<28s} shape={shp} dtype={getattr(arr, 'dtype', '?')}")
            print()
            break


def main():
    if MODE == "probe":
        probe()
        return
    print("rendering Tier 2 figures ->", FIGS_DIR)
    fig_t2_combined()
    fig_success_curves()
    fig_beta_trajectories()
    fig_occupancy()


if __name__ == "__main__":
    main()