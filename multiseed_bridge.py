import os
import sys
import glob
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTDIR = os.path.join(os.getcwd(), "runs", "bridge_multiseed")
os.makedirs(OUTDIR, exist_ok=True)

from exp_bridge import build_local_view, collect_critic_samples, occ_weighted_fidelity, _mix, GAE_LAMBDA, _torch_device
from sampled_apparatus import SampledApparatus
from apparatus import FeatureMap
from rollout import collect_rollouts, empirical_occupancy
from exp_control import build, TAU, LR, DELTA_C, DELTA_PHI, D_FEAT, J_OPT
from recording import save_run, load_run

HELD_BETAS = (0.2, 0.3, 0.4, 0.5)
DEFAULT_SEEDS = 5
SNAPSHOTS = (250, 800, 1600, 2499)
N_ITERS = 2500
TRAIN_EPS = 200
PROBE_EPS = 1500
CRITIC_ITERS = 40
RECIPE = dict(hidden=128, epochs=200, conv=(16, 32), lr=3e-3, batch=512, crop=7)
ROSTER = ("flat_nobeta", "onehot_nobeta", "conv_nobeta", "tabular")
FRAGILE = 0.02
DEVICE = _torch_device()


class FFQCritic:
    def __init__(self, env, A, *, encoder, conv=(16, 32), hidden=128, seed=0, lr=3e-3,
                 epochs=200, batch=512, crop=7, device="cpu"):
        if not _TORCH:
            raise RuntimeError("FFQCritic needs torch (run on the GPU box)")
        torch.manual_seed(seed)
        self.env, self.A, self.encoder = env, int(A), encoder
        self.device = torch.device(device)
        S = env.n_states
        if encoder == "onehot":
            X = np.eye(S, dtype=np.float32); self.enc = nn.Identity(); enc_out = S
        elif encoder == "flat":
            X = build_local_view(env, crop).reshape(S, -1).astype(np.float32)
            self.enc = nn.Identity(); enc_out = X.shape[1]
        elif encoder == "conv":
            X = build_local_view(env, crop); ch = X.shape[1]
            layers, cin = [], ch
            for cout in conv:
                layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]; cin = cout
            layers += [nn.Flatten()]
            self.enc = nn.Sequential(*layers); enc_out = cin * crop * crop
        else:
            raise ValueError(encoder)
        self.enc = self.enc.to(self.device)
        self.head = nn.Sequential(nn.Linear(enc_out, hidden), nn.ReLU(), nn.Linear(hidden, self.A)).to(self.device)
        self.Xt = torch.tensor(X, device=self.device)
        self.lr, self.epochs, self.batch = lr, epochs, batch

    def fit(self, s_idx, a_idx, beta_vals, targets):
        s = torch.as_tensor(np.asarray(s_idx), dtype=torch.long, device=self.device)
        a = torch.as_tensor(np.asarray(a_idx), dtype=torch.long, device=self.device)
        y = torch.as_tensor(np.asarray(targets, np.float32), device=self.device)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.head.parameters()), lr=self.lr)
        N = len(s)
        for _ in range(self.epochs):
            p = torch.randperm(N, device=self.device)
            for st in range(0, N, self.batch):
                bi = p[st:st + self.batch]
                out = self.head(self.enc(self.Xt[s[bi]]))
                pred = out.gather(1, a[bi][:, None]).squeeze(1)
                loss = ((pred - y[bi]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def Q(self, beta):
        with torch.no_grad():
            s = torch.arange(self.env.n_states, device=self.device)
            return self.head(self.enc(self.Xt[s])).cpu().numpy()

    def advantage(self, beta, pi_phi):
        Q = self.Q(beta)
        return Q - (pi_phi * Q).sum(axis=1, keepdims=True)


class TabularQ:
    def __init__(self, env, A):
        self.S, self.A = env.n_states, int(A)

    def fit(self, s_idx, a_idx, beta_vals, targets):
        Q = np.zeros((self.S, self.A)); cnt = np.zeros((self.S, self.A))
        np.add.at(Q, (np.asarray(s_idx), np.asarray(a_idx)), np.asarray(targets, float))
        np.add.at(cnt, (np.asarray(s_idx), np.asarray(a_idx)), 1.0)
        self.Qsa = np.where(cnt > 0, Q / np.maximum(cnt, 1), 0.0)
        return self

    def advantage(self, beta, pi_phi):
        return self.Qsa - (pi_phi * self.Qsa).sum(axis=1, keepdims=True)


def build_roster_critic(name, env, seed):
    A = env.n_actions
    if name == "tabular":
        return TabularQ(env, A)
    enc = {"flat_nobeta": "flat", "onehot_nobeta": "onehot", "conv_nobeta": "conv"}[name]
    return FFQCritic(env, A, encoder=enc, conv=RECIPE["conv"], hidden=RECIPE["hidden"],
                     seed=seed, lr=RECIPE["lr"], epochs=RECIPE["epochs"], batch=RECIPE["batch"],
                     crop=RECIPE["crop"], device=DEVICE)


def pair_name(beta, seed):
    return f"b{int(round(beta*100)):02d}_s{seed}"


def run_pair(held_beta, seed, env, panel, room, verbose=True):
    import time
    S, A, G = env.n_states, env.n_actions, env.gamma
    fm = FeatureMap.random(S, D_FEAT, seed=100 + seed)
    ap = SampledApparatus(panel, env, fm, A, advisor_mode="shaped", beta=held_beta, lam=1.0,
                          tau=TAU, lr=LR, delta_c=DELTA_C, delta_phi=DELTA_PHI,
                          distractor_states=room, seed=seed, n_episodes=TRAIN_EPS,
                          gae_lambda=GAE_LAMBDA, critic_iters=CRITIC_ITERS)
    snaps, last, t0 = {}, 0, time.time()
    for cp in sorted(SNAPSHOTS):
        if cp > last:
            ap.run(cp - last); last = cp
        snaps[cp] = (ap.polA.probs().copy(),
                     ap.polB.probs().copy() if ap.polB is not None
                     else ap._advisor_probs(ap.polA.probs(), held_beta, 1.0).copy())
        if verbose:
            print(f"    [{pair_name(held_beta,seed)}] snapshot {cp} | {(time.time()-t0)/60:.0f} min", flush=True)

    rows, arrays = [], {}
    for cp, (piA, piB) in snaps.items():
        piphi = _mix(piA, piB, held_beta)
        Aexact = panel.solve(piphi).A_ext
        ex = panel.control_gradient(piA, piB, held_beta, tau=TAU).dJ_dbeta
        sE = collect_critic_samples(env, piA, piB, [round(held_beta, 4)], "ext",
                                    n_episodes=PROBE_EPS, gae_lambda=GAE_LAMBDA,
                                    critic_iters=CRITIC_ITERS, seed=seed)
        dh = empirical_occupancy(collect_rollouts(env, piphi, PROBE_EPS, seed=seed + 7), S, G)
        arrays[f"piA_it{cp}"] = piA; arrays[f"piB_it{cp}"] = piB
        arrays[f"Aexact_it{cp}"] = Aexact; arrays[f"dhat_it{cp}"] = dh
        for name in ROSTER:
            c = build_roster_critic(name, env, seed); c.fit(*sE)
            Ah = c.advantage(held_beta, piphi)
            d = -float(dh @ ((piA - piB) * Ah).sum(1)) / (1 - G)
            occ, _, _ = occ_weighted_fidelity(panel, c, piA, piB, held_beta)
            rc = float(np.corrcoef(Ah.ravel(), Aexact.ravel())[0, 1])
            rows.append(dict(held_beta=held_beta, seed=seed, stage=cp, critic=name,
                             dJdb_exact=ex, dJdb_hat=d, sign_ok=int((d < 0) == (ex < 0)),
                             occ_corr=occ, raw_corr=rc, mag_ratio=abs(d) / (abs(ex) + 1e-12),
                             exact_abs=abs(ex)))
            arrays[f"A_{name}_it{cp}"] = Ah
            if verbose:
                print(f"      it{cp} {name:14s} exact={ex:+.4f} hat={d:+.4f} "
                      f"sign={'T' if (d<0)==(ex<0) else 'F'} occ={occ:+.2f} raw={rc:+.2f}")

    cfg = dict(experiment="bridge_multiseed", held_beta=held_beta, seed=seed, n_iters=N_ITERS,
               snapshots=list(SNAPSHOTS), train_eps=TRAIN_EPS, probe_eps=PROBE_EPS,
               critic_iters=CRITIC_ITERS, roster=list(ROSTER), recipe=str(RECIPE),
               gae_lambda=GAE_LAMBDA, tau=TAU, gamma=G, j_opt=J_OPT, fragile_thresh=FRAGILE,
               buffer="held-beta-only (no pooling, no beta-conditioning)",
               cnn_channels="wallN,wallE,wallS,wallW,agent_center,oob,goal_tile_identity",
               torch=torch.__version__, device=DEVICE, instrument="room-field (exp_control.build)")
    nm = pair_name(held_beta, seed)
    save_run(OUTDIR, nm + "_train", cfg, ap.log, arrays=arrays)
    save_run(OUTDIR, nm + "_probe", cfg, rows)
    return rows


def run_multiseed(seeds, verbose=True):
    env, panel, room = build()
    done = {os.path.basename(f).replace("_probe__log.csv", "")
            for f in glob.glob(os.path.join(OUTDIR, "*_probe__log.csv"))}
    todo = [(b, s) for b in HELD_BETAS for s in range(seeds)]
    print(f"multi-seed: {len(todo)} pairs ({len(HELD_BETAS)} betas x {seeds} seeds) | "
          f"{len(done)} already done | device={DEVICE} -> {OUTDIR}")
    for i, (b, s) in enumerate(todo):
        if pair_name(b, s) in done:
            print(f"  skip {pair_name(b,s)} (already saved)"); continue
        print(f"  [{i+1}/{len(todo)}] running {pair_name(b,s)} ...", flush=True)
        run_pair(b, s, env, panel, room, verbose=verbose)


def aggregate():
    import csv, collections
    files = sorted(glob.glob(os.path.join(OUTDIR, "*_probe__log.csv")))
    if not files:
        print("no pair files found; run the pairs first."); return
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f)))
    by = collections.defaultdict(list)
    for r in rows:
        by[(float(r["held_beta"]), int(r["stage"]), r["critic"])].append(r)
    seeds_seen = sorted({int(r["seed"]) for r in rows})
    print(f"\n=== MULTI-SEED SIGN-RELIABILITY ({len(seeds_seen)} seeds: {seeds_seen}) ===")
    print("  per cell: fraction of seeds with correct dial sign | mean occ-corr | exact |dJ/db| range")
    print("  * = cell where median exact |dJ/db| < {:.2f} (intrinsically near-zero / crossing)".format(FRAGILE))
    for crit in ROSTER:
        print(f"\n  critic: {crit}")
        print("   beta  stage |  sign-rate  | mean-occ |  exact|dJ/db| med (min..max)")
        for b in HELD_BETAS:
            for st in SNAPSHOTS:
                g = by.get((b, st, crit), [])
                if not g:
                    continue
                sr = np.mean([int(x["sign_ok"]) for x in g])
                oc = np.mean([float(x["occ_corr"]) for x in g])
                ea = np.array([float(x["exact_abs"]) for x in g])
                frag = " *" if np.median(ea) < FRAGILE else "  "
                print(f"   {b:.2f}  {st:5d} |  {sr*100:5.0f}%{'':3s}| {oc:+.2f}    | "
                      f"{np.median(ea):.4f} ({ea.min():.4f}..{ea.max():.4f}){frag}")
    res = [r for r in rows if r["critic"] == "flat_nobeta" and float(r["exact_abs"]) >= FRAGILE]
    if res:
        rate = np.mean([int(r["sign_ok"]) for r in res])
        print(f"\n  HEADLINE  flat_nobeta sign-rate on RESOLVABLE cells (|dJ/db|>={FRAGILE}): "
              f"{rate*100:.1f}%  ({sum(int(r['sign_ok']) for r in res)}/{len(res)} cells x seeds)")
    print("\n  (fragile cells reported but excluded from the headline -- their sign is near-zero by"
          "\n   construction, not a critic failure.)")


if __name__ == "__main__":
    seeds = DEFAULT_SEEDS
    if "--seeds" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--seeds") + 1])
    if "aggregate" in sys.argv:
        aggregate()
    else:
        run_multiseed(seeds)
        aggregate()