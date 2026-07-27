import argparse
import time
import json
import os
import csv
import sys
import subprocess
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from tier2_layer0_env_trap import FourRoomsTrapEnv, LeanEncoder, DetailEncoder

def run_stem(arm, size, lam, seed, tag=""):
    return f"{tag}{arm}_size{size}_lam{_fmt(lam)}_seed{seed}"

def _fmt(x):
    return f"{x:g}".replace(".", "p")

def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def write_meta(path, config, wall_s, extra=None):
    meta = dict(config=config, wall_s=wall_s,
                versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                              numpy=np.__version__,
                              gymnasium=gym.__version__,
                              minigrid=__import__("minigrid").__version__),
                git_commit=_git_commit(),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    if extra:
        meta.update(extra)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

CSV_FIELDS = ["step", "ext_return", "success_rate", "steps_to_goal", "inside_rnd",
              "v_loss", "approx_kl", "clip_frac", "lam_now",
              "deployed_occ_entropy", "deployed_goal_cell_occ", "deployed_goal_region_occ",
              "wall_s"]


class OccupancyTrackerWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99, goal_radius=2):
        super().__init__(env)
        self._mg = env.unwrapped
        self.gamma = gamma
        self.observation_space = env.observation_space
        self._goal_cell = None
        self._goal_radius = goal_radius
        W, H = self._mg.width, self._mg.height
        self.visit_counts = np.zeros((W, H), dtype=np.int64)
        self.disc_occ = np.zeros((W, H), dtype=np.float64)
        self._win_disc_occ = np.zeros((W, H), dtype=np.float64)
        self._win_visit = np.zeros((W, H), dtype=np.int64)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._disc = 1.0
        self.goal_occ = 0.0
        if self._goal_cell is None:
            self._goal_cell = self._find_goal()
        self._mark(tuple(self._mg.agent_pos))
        return obs, self._aug(info)

    def step(self, a):
        obs, rew, term, trunc, info = self.env.step(a)
        pos = tuple(self._mg.agent_pos)
        self._mark(pos)
        if self._goal_cell is not None and pos == self._goal_cell:
            self.goal_occ += self._disc
        self._disc *= self.gamma
        return obs, rew, term, trunc, self._aug(info)

    def _mark(self, pos):
        x, y = pos
        if 0 <= x < self.visit_counts.shape[0] and 0 <= y < self.visit_counts.shape[1]:
            self.visit_counts[x, y] += 1
            self.disc_occ[x, y] += (1.0 - self.gamma) * self._disc
            self._win_visit[x, y] += 1
            self._win_disc_occ[x, y] += (1.0 - self.gamma) * self._disc

    def _aug(self, info):
        info = dict(info)
        info["goal_occ"] = (1.0 - self.gamma) * self.goal_occ
        info["agent_pos"] = tuple(self._mg.agent_pos)
        return info

    def occupancy_maps(self):
        total = self.disc_occ.sum()
        d = self.disc_occ / total if total > 0 else self.disc_occ
        return self.visit_counts.copy(), d

    def snapshot_and_reset(self):
        total = self._win_disc_occ.sum()
        d = self._win_disc_occ / total if total > 0 else self._win_disc_occ.copy()
        nz = d[d > 0]
        entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
        goal_cell_occ = 0.0
        goal_region_occ = 0.0
        if self._goal_cell is not None:
            gx, gy = self._goal_cell
            goal_cell_occ = float(d[gx, gy])
            r = self._goal_radius
            x0, x1 = max(0, gx - r), min(d.shape[0], gx + r + 1)
            y0, y1 = max(0, gy - r), min(d.shape[1], gy + r + 1)
            goal_region_occ = float(d[x0:x1, y0:y1].sum())
        scalars = dict(entropy=entropy, goal_cell_occ=goal_cell_occ,
                       goal_region_occ=goal_region_occ, window_visits=int(self._win_visit.sum()))
        win_visit = self._win_visit.copy()
        self._win_disc_occ.fill(0.0)
        self._win_visit.fill(0)
        return d, win_visit, scalars

    def _find_goal(self):
        for i in range(self._mg.width):
            for j in range(self._mg.height):
                c = self._mg.grid.get(i, j)
                if c is not None and c.type == "goal":
                    return (i, j)
        return None


def make_env(size=13, max_steps=2000, gamma=0.99, tile_size=8):
    from minigrid.wrappers import RGBImgPartialObsWrapper
    rh = size // 2
    goal = (rh + (size - 1 - rh) // 2, 1 + (rh - 1) // 2)
    base = FourRoomsTrapEnv(size=size, agent_pos=(1, 1), goal_pos=goal,
                            max_steps=max_steps, render_mode="rgb_array")
    rgb = RGBImgPartialObsWrapper(base, tile_size=tile_size)
    return OccupancyTrackerWrapper(rgb, gamma=gamma)


def make_doorkey_env(size=8, max_steps=2000, gamma=0.99, tile_size=8):
    from minigrid.wrappers import RGBImgPartialObsWrapper
    from minigrid.envs import DoorKeyEnv
    base = DoorKeyEnv(size=size, max_steps=max_steps, render_mode="rgb_array")
    rgb = RGBImgPartialObsWrapper(base, tile_size=tile_size)
    return OccupancyTrackerWrapper(rgb, gamma=gamma)


class RND(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.target = DetailEncoder(feat_dim=feat_dim)
        self.predictor = DetailEncoder(feat_dim=feat_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self._run_M2 = 0.0
        self._run_mean = 0.0
        self._count = 1e-4

    def raw_error(self, obs):
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        return ((p - t) ** 2).mean(dim=1)

    def intrinsic_reward(self, obs):
        with torch.no_grad():
            e = self.raw_error(obs)
            for v in e.detach().cpu().numpy().ravel():
                self._count += 1.0
                delta = v - self._run_mean
                self._run_mean += delta / self._count
                self._run_M2 += delta * (v - self._run_mean)
            var = self._run_M2 / max(self._count - 1.0, 1.0)
            std = float(np.sqrt(var) + 1e-8)
            return e / std

    def distill_loss(self, obs):
        return self.raw_error(obs).mean()


class RBAgent(nn.Module):
    def __init__(self, n_actions, feat_dim=256):
        super().__init__()
        self.encoder = LeanEncoder(feat_dim=feat_dim)
        self.actor = nn.Linear(feat_dim, n_actions)
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        f = self.encoder(obs)
        return self.actor(f), self.critic(f).squeeze(-1)

    def act(self, obs):
        logits, v = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), v


def to_obs_tensor(img, device):
    return torch.as_tensor(np.asarray(img), device=device).unsqueeze(0)


def train_rb(size=13, total_steps=60000, max_steps=2000, rollout=512, epochs=4, minibatch=128,
             lr=2.5e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
             max_grad_norm=0.5, lr_anneal_floor=1.0,
             lam=1.0, lambda_anneal_start=0.5, rnd_lr=1e-4, device="cpu", seed=0, log_every=2000, runs_dir="runs", tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    config = dict(arm="rbanneal", lambda_anneal_start=lambda_anneal_start, size=size, total_steps=total_steps, max_steps=max_steps, lam=lam,
                  seed=seed, rollout=rollout, epochs=epochs, minibatch=minibatch, lr=lr, gamma=gamma,
                  gae_lambda=gae_lambda, clip=clip, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, rnd_lr=rnd_lr, lr_anneal_floor=lr_anneal_floor)
    arm_dir = os.path.join(runs_dir, "tier2", "rbanneal")
    os.makedirs(arm_dir, exist_ok=True)
    stem = run_stem("rbanneal", size, lam, seed, tag)
    csv_path = os.path.join(arm_dir, stem + ".csv")
    meta_path = os.path.join(arm_dir, stem + "_meta.json")
    occ_path = os.path.join(arm_dir, stem + "_occ.npz")
    ep_path = os.path.join(arm_dir, stem + "_episodes.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS); csv_w.writeheader()
    EP_FIELDS = ["episode_idx", "end_step", "length", "ext_return", "int_return",
                 "terminated", "source", "beta"]
    ep_f = open(ep_path, "w", newline="")
    ep_w = csv.DictWriter(ep_f, fieldnames=EP_FIELDS); ep_w.writeheader()
    ep_counter = 0
    print(f"logging -> {csv_path}\n           {ep_path}")

    env = make_env(size=size, max_steps=max_steps, gamma=gamma)
    n_act = env.action_space.n
    agent = RBAgent(n_act).to(device)
    rnd = RND().to(device)
    obs, info = env.reset(seed=seed)
    o = to_obs_tensor(obs["image"], device)
    agent.act(o); rnd.intrinsic_reward(o)
    opt = torch.optim.Adam(list(agent.parameters()), lr=lr)
    rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=rnd_lr)

    t0 = time.time()
    global_step = 0
    ep_ext_return = 0.0
    ep_int_return = 0.0
    recent_ext = []
    recent_int = []
    recent_success = []
    recent_steps = []
    ep_len = 0
    snap_maps, snap_visits, snap_steps = [], [], []
    snap_entropy, snap_goal_cell, snap_goal_region = [], [], []
    obs, info = env.reset(seed=seed)

    while global_step < total_steps:
        frac = global_step / max(total_steps, 1)
        if frac <= lambda_anneal_start:
            lam_now = lam
        else:
            _p = (frac - lambda_anneal_start) / max(1.0 - lambda_anneal_start, 1e-8)
            lam_now = lam * max(0.0, 1.0 - _p)
        lr_now = lr * (1.0 - frac * (1.0 - lr_anneal_floor))
        for g in opt.param_groups:
            g["lr"] = lr_now
        obs_buf, act_buf, logp_buf, val_buf, done_buf = [], [], [], [], []
        rew_ext_buf, rew_int_buf = [], []
        term_buf = []
        next_val_buf = []
        rnd_obs_buf = []
        for _ in range(rollout):
            o = to_obs_tensor(obs["image"], device)
            with torch.no_grad():
                a, logp, _, v = agent.act(o)
            nobs, r_ext, term, trunc, info = env.step(int(a.item()))
            with torch.no_grad():
                no = to_obs_tensor(nobs["image"], device)
                r_int = float(rnd.intrinsic_reward(no).item())
                if term:
                    next_v = 0.0
                else:
                    _, _nv = agent.forward(no)
                    next_v = float(_nv.item())
            done = term or trunc
            obs_buf.append(obs["image"]); act_buf.append(int(a.item()))
            logp_buf.append(float(logp.item())); val_buf.append(float(v.item()))
            next_val_buf.append(next_v)
            rew_ext_buf.append(float(r_ext)); rew_int_buf.append(float(r_int))
            done_buf.append(done)
            term_buf.append(bool(term))
            rnd_obs_buf.append(nobs["image"])
            recent_int.append(r_int)
            ep_ext_return += float(r_ext); ep_int_return += float(r_int)
            ep_len += 1
            global_step += 1
            obs = nobs
            if done:
                success = 1 if term else 0
                recent_ext.append(ep_ext_return)
                ep_w.writerow(dict(episode_idx=ep_counter, end_step=global_step, length=ep_len,
                                   ext_return=ep_ext_return, int_return=ep_int_return,
                                   terminated=success, source="deploy", beta=float("nan")))
                ep_counter += 1
                ep_ext_return = 0.0; ep_int_return = 0.0
                recent_success.append(success)
                if success:
                    recent_steps.append(ep_len)
                ep_len = 0
                obs, info = env.reset()

        win_d, win_v, win_sc = env.snapshot_and_reset()
        snap_maps.append(win_d.astype(np.float32)); snap_visits.append(win_v.astype(np.int32))
        snap_steps.append(global_step)
        snap_entropy.append(win_sc["entropy"]); snap_goal_cell.append(win_sc["goal_cell_occ"])
        snap_goal_region.append(win_sc["goal_region_occ"])

        row_task = dict(step=global_step,
                        ext_return=float(np.mean(recent_ext[-20:])) if recent_ext else 0.0,
                        success_rate=float(np.mean(recent_success[-20:])) if recent_success else 0.0,
                        steps_to_goal=float(np.mean(recent_steps[-20:])) if recent_steps else float("nan"),
                        inside_rnd=float(np.mean(recent_int[-1000:])) if recent_int else float("nan"),
                        deployed_occ_entropy=win_sc["entropy"],
                        deployed_goal_cell_occ=win_sc["goal_cell_occ"],
                        deployed_goal_region_occ=win_sc["goal_region_occ"],
                        wall_s=time.time() - t0)
        if global_step % log_every < rollout:
            print(f"  step {global_step:6d} | ext_ret {row_task['ext_return']:.3f} | "
                  f"succ {row_task['success_rate']:.2f} | steps {row_task['steps_to_goal']:.0f} | "
                  f"int_R {row_task['inside_rnd']:.2f} | {row_task['wall_s']:.0f}s")


        rew_buf = [re + lam_now * ri for re, ri in zip(rew_ext_buf, rew_int_buf)]

        adv = np.zeros(len(rew_buf), dtype=np.float32); gae = 0.0
        for t in reversed(range(len(rew_buf))):
            bootstrap_nonterm = 1.0 - float(term_buf[t])
            carry_nonterm = 1.0 - float(done_buf[t])
            delta = rew_buf[t] + gamma * next_val_buf[t] * bootstrap_nonterm - val_buf[t]
            gae = delta + gamma * gae_lambda * carry_nonterm * gae
            adv[t] = gae
        ret = adv + np.asarray(val_buf, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ob = torch.as_tensor(np.asarray(obs_buf), device=device)
        ac = torch.as_tensor(np.asarray(act_buf), device=device)
        oldlogp = torch.as_tensor(np.asarray(logp_buf), device=device)
        advt = torch.as_tensor(adv, device=device)
        rett = torch.as_tensor(ret, device=device)
        rnd_ob = torch.as_tensor(np.asarray(rnd_obs_buf), device=device)
        _ret = np.asarray(ret, dtype=np.float64); _vb = np.asarray(val_buf, dtype=np.float64)
        _ra = np.asarray(adv, dtype=np.float64)
        diag_ret_mean, diag_ret_std = float(_ret.mean()), float(_ret.std())
        diag_vbuf_mean, diag_vbuf_std = float(_vb.mean()), float(_vb.std())
        diag_rawadv_absmean = float(np.abs(_ra).mean())
        idx = np.arange(len(rew_buf))
        _vl_acc, _kl_acc, _cf_acc, _n_upd = 0.0, 0.0, 0.0, 0
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), minibatch):
                mb = idx[s:s + minibatch]
                logits, v = agent.forward(ob[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(ac[mb]); ent = dist.entropy().mean()
                ratio = torch.exp(lp - oldlogp[mb])
                pg1 = ratio * advt[mb]
                pg2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advt[mb]
                pol_loss = -torch.min(pg1, pg2).mean()
                v_loss = ((v - rett[mb]) ** 2).mean()
                loss = pol_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                opt.step()
                rnd_opt.zero_grad(); rnd.distill_loss(rnd_ob[mb]).backward(); rnd_opt.step()
                with torch.no_grad():
                    _vl_acc += float(v_loss.item())
                    _kl_acc += float((oldlogp[mb] - lp).mean().item())
                    _cf_acc += float(((ratio - 1.0).abs() > clip).float().mean().item())
                    _n_upd += 1
        _nu = max(_n_upd, 1)
        row = dict(row_task,
                   v_loss=_vl_acc / _nu, approx_kl=_kl_acc / _nu, clip_frac=_cf_acc / _nu,
                   lam_now=lam_now)
        csv_w.writerow(row); csv_f.flush()
        if global_step % log_every < rollout:
            print(f"           diag | v_loss {_vl_acc/_nu:.4f} | "
                  f"approx_kl {_kl_acc/_nu:+.4f} | clip_frac {_cf_acc/_nu:.3f} | "
                  f"lr {lr_now:.2e}")
            print(f"           mags | ret {diag_ret_mean:+.4f}+/-{diag_ret_std:.4f} | "
                  f"vbuf {diag_vbuf_mean:+.4f}+/-{diag_vbuf_std:.4f} | "
                  f"raw|adv| {diag_rawadv_absmean:.4f}")

    dt = time.time() - t0
    csv_f.close(); ep_f.close()
    visit, docc = env.occupancy_maps()
    np.savez(occ_path, visit_counts=visit, disc_occ=docc, arm="rb", lam=lam, seed=seed,
             deployed_occ_snapshots=np.asarray(snap_maps, dtype=np.float32),
             deployed_occ_snapshot_visits=np.asarray(snap_visits, dtype=np.int32),
             deployed_occ_snapshot_steps=np.asarray(snap_steps, dtype=np.int64),
             deployed_snap_entropy=np.asarray(snap_entropy, dtype=np.float32),
             deployed_snap_goal_cell_occ=np.asarray(snap_goal_cell, dtype=np.float32),
             deployed_snap_goal_region_occ=np.asarray(snap_goal_region, dtype=np.float32))
    config["n_actions"] = int(n_act)
    write_meta(meta_path, config, dt, extra=dict(ms_per_step=1000 * dt / total_steps,
                                                 occ_file=os.path.basename(occ_path)))
    print(f"\nRB run done | size={size} steps={total_steps} lam={lam} seed={seed} | "
          f"{dt:.1f}s ({1000*dt/total_steps:.2f} ms/step) [TIMING]")
    print(f"  curve -> {csv_path}\n  eps   -> {ep_path}\n  meta  -> {meta_path}\n  occ   -> {occ_path}")
    return dict(csv=csv_path, meta=meta_path, occ=occ_path, wall_s=dt)


if __name__ == "__main__":
    SIZE = 13
    STEPS = 100000
    MAX_STEPS = 1024
    LAMBDAS = [0.05]
    LAMBDA_ANNEAL_START = 0.5
    SEEDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ROLLOUT = 512
    EPOCHS = 4
    MINIBATCH = 128
    LR = 2.5e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.975
    CLIP = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    LR_ANNEAL_FLOOR = 0.1
    RND_LR = 1e-4
    DEVICE = "cuda"
    RUNS_DIR = "runs"
    TAG = "rbanneal_"
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS)
    ap.add_argument("--lam", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--lambda_anneal_start", type=float, default=LAMBDA_ANNEAL_START)
    ap.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--rollout", type=int, default=ROLLOUT)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--minibatch", type=int, default=MINIBATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--gae_lambda", type=float, default=GAE_LAMBDA)
    ap.add_argument("--clip", type=float, default=CLIP)
    ap.add_argument("--ent_coef", type=float, default=ENT_COEF)
    ap.add_argument("--vf_coef", type=float, default=VF_COEF)
    ap.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    ap.add_argument("--lr_anneal_floor", type=float, default=LR_ANNEAL_FLOOR)
    ap.add_argument("--rnd_lr", type=float, default=RND_LR)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--runs_dir", default=RUNS_DIR)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [device] cuda requested but unavailable -> falling back to cpu")
        args.device = "cpu"
    print(f"sweep: lambdas={args.lam} seeds={args.seed} | size={args.size} steps={args.steps} "
          f"max_steps={args.max_steps} | lr={args.lr} anneal_floor={args.lr_anneal_floor} "
          f"ent={args.ent_coef} clip={args.clip} vf={args.vf_coef} gnorm={args.max_grad_norm} "
          f"gae_lam={args.gae_lambda} rnd_lr={args.rnd_lr} | device={args.device} | tag='{args.tag}'")
    for lam in args.lam:
        for seed in args.seed:
            print(f"\n=== RB | lam={lam} seed={seed} ===")
            train_rb(size=args.size, total_steps=args.steps, max_steps=args.max_steps,
                     rollout=args.rollout, epochs=args.epochs, minibatch=args.minibatch,
                     lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda, clip=args.clip,
                     ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                     lr_anneal_floor=args.lr_anneal_floor, lam=lam,
                     lambda_anneal_start=args.lambda_anneal_start, rnd_lr=args.rnd_lr,
                     seed=seed, device=args.device, runs_dir=args.runs_dir, tag=args.tag)