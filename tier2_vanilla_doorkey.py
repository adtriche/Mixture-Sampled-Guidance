import argparse
import time
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from tier2_layer0_env_trap import LeanEncoder
from tier2_rb_learner import make_doorkey_env as make_env, to_obs_tensor, run_stem, write_meta, CSV_FIELDS


class VanillaAgent(nn.Module):
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


def train_vanilla(size=13, total_steps=60000, max_steps=2000, rollout=512, epochs=4, minibatch=128,
                  lr=2.5e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
                  max_grad_norm=0.5, lr_anneal_floor=1.0,
                  device="cpu", seed=0, log_every=2000, runs_dir="runs", tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    config = dict(arm="vanilla", size=size, total_steps=total_steps, max_steps=max_steps, seed=seed,
                  rollout=rollout, epochs=epochs, minibatch=minibatch, lr=lr, gamma=gamma,
                  gae_lambda=gae_lambda, clip=clip, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr_anneal_floor=lr_anneal_floor)
    arm_dir = os.path.join(runs_dir, "tier2", "vanilla")
    os.makedirs(arm_dir, exist_ok=True)
    stem = run_stem("vanilla", size, 0.0, seed, tag)
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
    agent = VanillaAgent(n_act).to(device)
    obs, info = env.reset(seed=seed)
    agent.act(to_obs_tensor(obs["image"], device))
    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    t0 = time.time()
    global_step = 0
    ep_ext_return = 0.0; ep_len = 0
    recent_ext = []; recent_success = []; recent_steps = []
    snap_maps, snap_visits, snap_steps = [], [], []
    snap_entropy, snap_goal_cell, snap_goal_region = [], [], []
    obs, info = env.reset(seed=seed)

    while global_step < total_steps:
        frac = global_step / max(total_steps, 1)
        lr_now = lr * (1.0 - frac * (1.0 - lr_anneal_floor))
        for g in opt.param_groups:
            g["lr"] = lr_now
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        term_buf = []
        next_val_buf = []
        for _ in range(rollout):
            o = to_obs_tensor(obs["image"], device)
            with torch.no_grad():
                a, logp, _, v = agent.act(o)
            nobs, r_ext, term, trunc, info = env.step(int(a.item()))
            done = term or trunc
            with torch.no_grad():
                if term:
                    next_v = 0.0
                else:
                    _, _nv = agent.forward(to_obs_tensor(nobs["image"], device))
                    next_v = float(_nv.item())
            obs_buf.append(obs["image"]); act_buf.append(int(a.item()))
            logp_buf.append(float(logp.item())); val_buf.append(float(v.item()))
            next_val_buf.append(next_v)
            rew_buf.append(float(r_ext))
            done_buf.append(done)
            term_buf.append(bool(term))
            ep_ext_return += float(r_ext); ep_len += 1
            global_step += 1
            obs = nobs
            if done:
                success = 1 if term else 0
                recent_ext.append(ep_ext_return)
                ep_w.writerow(dict(episode_idx=ep_counter, end_step=global_step, length=ep_len,
                                   ext_return=ep_ext_return, int_return=float("nan"),
                                   terminated=success, source="deploy", beta=float("nan")))
                ep_counter += 1
                ep_ext_return = 0.0
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
                        inside_rnd=float("nan"),
                        deployed_occ_entropy=win_sc["entropy"],
                        deployed_goal_cell_occ=win_sc["goal_cell_occ"],
                        deployed_goal_region_occ=win_sc["goal_region_occ"],
                        wall_s=time.time() - t0)
        if global_step % log_every < rollout:
            print(f"  step {global_step:6d} | ext_ret {row_task['ext_return']:.3f} | "
                  f"succ {row_task['success_rate']:.2f} | steps {row_task['steps_to_goal']:.0f} | "
                  f"{row_task['wall_s']:.0f}s")

        adv = np.zeros(len(rew_buf), dtype=np.float32); gae = 0.0
        for t in reversed(range(len(rew_buf))):
            bootstrap_nonterm = 1.0 - float(term_buf[t])
            carry_nonterm = 1.0 - float(done_buf[t])
            delta = rew_buf[t] + gamma * next_val_buf[t] * bootstrap_nonterm - val_buf[t]
            gae = delta + gamma * gae_lambda * carry_nonterm * gae
            adv[t] = gae
        ret = adv + np.asarray(val_buf, dtype=np.float32)

        _ret = np.asarray(ret, dtype=np.float64)
        _vb = np.asarray(val_buf, dtype=np.float64)
        _ra = np.asarray(adv, dtype=np.float64)
        diag_ret_mean, diag_ret_std = float(_ret.mean()), float(_ret.std())
        diag_vbuf_mean, diag_vbuf_std = float(_vb.mean()), float(_vb.std())
        diag_rawadv_absmean = float(np.abs(_ra).mean())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ob = torch.as_tensor(np.asarray(obs_buf), device=device)
        ac = torch.as_tensor(np.asarray(act_buf), device=device)
        oldlogp = torch.as_tensor(np.asarray(logp_buf), device=device)
        advt = torch.as_tensor(adv, device=device); rett = torch.as_tensor(ret, device=device)
        idx = np.arange(len(rew_buf))

        _vl_acc, _kl_acc, _cf_acc, _n_upd = 0.0, 0.0, 0.0, 0
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), minibatch):
                mb = idx[s:s + minibatch]
                logits, vv = agent.forward(ob[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(ac[mb]); ent = dist.entropy().mean()
                ratio = torch.exp(lp - oldlogp[mb])
                pg1 = ratio * advt[mb]; pg2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advt[mb]
                pol_loss = -torch.min(pg1, pg2).mean()
                v_loss = ((vv - rett[mb]) ** 2).mean()
                loss = pol_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                opt.step()
                with torch.no_grad():
                    _vl_acc += float(v_loss.item())
                    _kl_acc += float((oldlogp[mb] - lp).mean().item())
                    _cf_acc += float(((ratio - 1.0).abs() > clip).float().mean().item())
                    _n_upd += 1
        diag_vloss = _vl_acc / max(_n_upd, 1)
        diag_kl = _kl_acc / max(_n_upd, 1)
        diag_clipfrac = _cf_acc / max(_n_upd, 1)
        row = dict(row_task, v_loss=diag_vloss, approx_kl=diag_kl, clip_frac=diag_clipfrac)
        csv_w.writerow(row); csv_f.flush()
        if global_step % log_every < rollout:
            print(f"           diag | v_loss {diag_vloss:.4f} | approx_kl {diag_kl:+.4f} | "
                  f"clip_frac {diag_clipfrac:.3f} | lr {lr_now:.2e}")
            print(f"           mags | ret {diag_ret_mean:+.4f}+/-{diag_ret_std:.4f} | "
                  f"vbuf {diag_vbuf_mean:+.4f}+/-{diag_vbuf_std:.4f} | "
                  f"raw|adv| {diag_rawadv_absmean:.4f}")

    dt = time.time() - t0
    csv_f.close(); ep_f.close()
    visit, docc = env.occupancy_maps()
    np.savez(occ_path, visit_counts=visit, disc_occ=docc, arm="vanilla", lam=0.0, seed=seed,
             deployed_occ_snapshots=np.asarray(snap_maps, dtype=np.float32),
             deployed_occ_snapshot_visits=np.asarray(snap_visits, dtype=np.int32),
             deployed_occ_snapshot_steps=np.asarray(snap_steps, dtype=np.int64),
             deployed_snap_entropy=np.asarray(snap_entropy, dtype=np.float32),
             deployed_snap_goal_cell_occ=np.asarray(snap_goal_cell, dtype=np.float32),
             deployed_snap_goal_region_occ=np.asarray(snap_goal_region, dtype=np.float32))
    config["n_actions"] = int(n_act)
    write_meta(meta_path, config, dt, extra=dict(ms_per_step=1000 * dt / total_steps,
                                                 occ_file=os.path.basename(occ_path)))
    print(f"\nvanilla run done | size={size} steps={total_steps} seed={seed} | "
          f"{dt:.1f}s ({1000*dt/total_steps:.2f} ms/step) [TIMING]")
    print(f"  curve -> {csv_path}\n  eps   -> {ep_path}\n  meta  -> {meta_path}\n  occ   -> {occ_path}")
    return dict(csv=csv_path, meta=meta_path, occ=occ_path, wall_s=dt)


if __name__ == "__main__":
    SIZE = 6
    STEPS = 100000
    MAX_STEPS = 1024
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
    DEVICE = "cuda"
    RUNS_DIR = "runs"
    TAG = "doorkey_"
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS)
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
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--runs_dir", default=RUNS_DIR)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [device] cuda requested but unavailable -> falling back to cpu")
        args.device = "cpu"
    print(f"vanilla: seeds={args.seed} | size={args.size} steps={args.steps} "
          f"max_steps={args.max_steps} | lr={args.lr} anneal_floor={args.lr_anneal_floor} "
          f"ent={args.ent_coef} clip={args.clip} vf={args.vf_coef} gnorm={args.max_grad_norm} "
          f"gae_lam={args.gae_lambda} | device={args.device} | tag='{args.tag}'")
    for seed in args.seed:
        print(f"\n=== vanilla | seed={seed} ===")
        train_vanilla(size=args.size, total_steps=args.steps, max_steps=args.max_steps,
                      rollout=args.rollout, epochs=args.epochs, minibatch=args.minibatch,
                      lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda, clip=args.clip,
                      ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                      lr_anneal_floor=args.lr_anneal_floor,
                      seed=seed, device=args.device, runs_dir=args.runs_dir, tag=args.tag)