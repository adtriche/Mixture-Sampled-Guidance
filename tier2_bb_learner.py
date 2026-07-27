import argparse
import time
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from tier2_layer0_env_trap import LeanEncoder
from tier2_rb_learner import (make_env, to_obs_tensor, run_stem, write_meta, RND)


class Policy(nn.Module):
    def __init__(self, n_actions, feat_dim=256):
        super().__init__()
        self.encoder = LeanEncoder(feat_dim=feat_dim)
        self.actor = nn.Linear(feat_dim, n_actions)
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        f = self.encoder(obs)
        return self.actor(f), self.critic(f).squeeze(-1)

    def logits(self, obs):
        return self.actor(self.encoder(obs))

    def value(self, obs):
        return self.critic(self.encoder(obs)).squeeze(-1)


def mixture_log_prob(logits_A, logits_B, beta, actions):
    pA = torch.softmax(logits_A, dim=-1)
    pB = torch.softmax(logits_B, dim=-1)
    pmix = (1.0 - beta) * pA + beta * pB
    p_a = pmix.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return torch.log(p_a + 1e-12), pmix


def statewise_tv(logits_A, logits_B):
    pA = torch.softmax(logits_A, dim=-1)
    pB = torch.softmax(logits_B, dim=-1)
    return 0.5 * (pA - pB).abs().sum(dim=-1)


def actor_eval(actor, make_env_fn, device, n_episodes=20, max_steps=2000, seed=10_000):
    env = make_env_fn()
    succ, steps = [], []
    ep_records = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False; t = 0; ep_ext = 0.0
        while not done:
            o = to_obs_tensor(obs["image"], device)
            with torch.no_grad():
                a = torch.distributions.Categorical(logits=actor.logits(o)).sample()
            obs, r, term, trunc, info = env.step(int(a.item()))
            ep_ext += float(r)
            done = term or trunc; t += 1
        succ.append(1 if term else 0)
        if term:
            steps.append(t)
        ep_records.append(dict(length=t, ext_return=ep_ext, terminated=1 if term else 0))
    _, docc = env.occupancy_maps()
    _adocc, _avis, ascalars = env.snapshot_and_reset()
    return (float(np.mean(succ)),
            float(np.mean(steps)) if steps else float("nan"),
            docc, ascalars, ep_records)


CSV_FIELDS = ["step", "mix_ext_return", "mix_success", "mix_steps",
              "actor_eval_success", "actor_eval_steps",
              "beta", "beta_pending", "G_beta", "H_lambda", "ev_vE", "rankcorr_vE",
              "rankcorr_qE_post", "qlossE_first", "qlossE_drop",
              "D_e", "I_e", "int_R",
              "ent_A", "ent_B", "wA_mean", "wB_mean", "actor_gradnorm",
              "withdraw_fired", "dbeta_raw", "dbeta_applied", "beta_pending_pre", "beta_pending_post",
              "approx_kl_A", "approx_kl_B", "clip_frac_A", "clip_frac_B", "v_loss_A", "v_loss_B",
              "tighten_frac",
              "mt2", "ct2", "r_Aphi_max", "frac_near0",
              "deployed_occ_entropy", "deployed_goal_cell_occ", "deployed_goal_region_occ",
              "actor_occ_entropy", "actor_goal_cell_occ", "actor_goal_region_occ",
              "wall_s"]


def train_bb(size=13, total_steps=60000, max_steps=2000, rollout=512, epochs=4, minibatch=128,
             lr=2.5e-4, gamma=0.99, gae_lambda=0.975, clip=0.2, ent_coef=0.1, ent_coef_end=0.01,
             ent_coef_B=0.1, vf_coef=0.5, max_grad_norm=0.5, lr_anneal_floor=1.0,
             delta_c=0.10, delta_phi=0.05,
             beta=0.45, lam=1.0, rnd_lr=1e-4, controller=False, delta_beta=0.075, d_beta_floor=1e-3,
             device="cpu", seed=0, log_every=2000, eval_every=10000, runs_dir="runs", tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    config = dict(arm="bb", layer=(2 if controller else 1), size=size, total_steps=total_steps,
                  max_steps=max_steps, controller=controller, delta_beta=delta_beta,
                  beta_init=beta,
                  beta=beta, lam=lam, seed=seed, rollout=rollout, epochs=epochs, minibatch=minibatch,
                  lr=lr, gamma=gamma, gae_lambda=gae_lambda, clip=clip, ent_coef=ent_coef, ent_coef_end=ent_coef_end, ent_coef_B=ent_coef_B,
                  vf_coef=vf_coef, rnd_lr=rnd_lr,
                  delta_c=delta_c, delta_phi=delta_phi, lr_anneal_floor=lr_anneal_floor)
    arm_dir = os.path.join(runs_dir, "tier2", "bb")
    os.makedirs(arm_dir, exist_ok=True)
    stem = run_stem("bb", size, lam, seed, tag)
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
    ep_counter = {"deploy": 0, "actor_eval": 0}
    print(f"logging -> {csv_path}\n           {ep_path}")

    def mk():
        return make_env(size=size, max_steps=max_steps, gamma=gamma)
    env = mk()
    n_act = env.action_space.n
    actor = Policy(n_act).to(device)
    advisor = Policy(n_act).to(device)
    rnd = RND().to(device)
    # lazy-init heads
    obs, info = env.reset(seed=seed)
    o = to_obs_tensor(obs["image"], device)
    actor.forward(o); advisor.forward(o); rnd.intrinsic_reward(o)
    opt_A = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_B = torch.optim.Adam(advisor.parameters(), lr=lr)
    rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=rnd_lr)

    t0 = time.time(); global_step = 0
    beta_pending = beta
    ep_ext = 0.0; ep_int = 0.0; ep_len = 0
    recent_ext = []; recent_succ = []; recent_steps = []; recent_int = []
    last_eval = dict(success=float("nan"), steps=float("nan"),
                     occ_entropy=float("nan"), goal_cell_occ=float("nan"),
                     goal_region_occ=float("nan"))
    mix_snap_maps, mix_snap_visits, mix_snap_steps = [], [], []
    mix_snap_entropy, mix_snap_goal_cell, mix_snap_goal_region = [], [], []
    actor_snap_maps, actor_snap_steps = [], []
    actor_snap_entropy, actor_snap_goal_cell, actor_snap_goal_region = [], [], []
    obs, info = env.reset(seed=seed)

    while global_step < total_steps:
        frac = min(1.0, global_step / max(total_steps, 1))
        ent_coef_A = ent_coef + frac * (ent_coef_end - ent_coef)
        lr_now = lr * (1.0 - frac * (1.0 - lr_anneal_floor))
        for _g in opt_A.param_groups: _g["lr"] = lr_now
        for _g in opt_B.param_groups: _g["lr"] = lr_now
        obs_buf, act_buf = [], []
        logp_mix_buf = []
        rew_ext_buf, rew_int_buf, done_buf = [], [], []
        term_buf = []
        vE_buf, vI_buf = [], []
        next_vE_buf, next_vI_buf = [], []
        De_states = []
        pA_buf, pB_buf, pmix_buf = [], [], []
        for _ in range(rollout):
            o = to_obs_tensor(obs["image"], device)
            with torch.no_grad():
                lA, vE = actor.forward(o)
                lB = advisor.logits(o); vI = advisor.value(o)
                pA = torch.softmax(lA, dim=-1); pB = torch.softmax(lB, dim=-1)
                pmix = (1.0 - beta) * pA + beta * pB
                a = torch.distributions.Categorical(probs=pmix).sample()
                lp_a = torch.log(pmix.gather(-1, a.unsqueeze(-1)).squeeze(-1) + 1e-12)
                tv = 0.5 * (pA - pB).abs().sum(dim=-1)
            nobs, r_ext, term, trunc, info = env.step(int(a.item()))
            with torch.no_grad():
                no = to_obs_tensor(nobs["image"], device)
                r_int = float(rnd.intrinsic_reward(no).item())
                if term:
                    next_vE = 0.0; next_vI = 0.0
                else:
                    _, next_vE_t = actor.forward(no); next_vI_t = advisor.value(no)
                    next_vE = float(next_vE_t.item()); next_vI = float(next_vI_t.item())
            done = term or trunc
            obs_buf.append(obs["image"]); act_buf.append(int(a.item()))
            logp_mix_buf.append(float(lp_a.item()))
            rew_ext_buf.append(float(r_ext)); rew_int_buf.append(r_int)
            vE_buf.append(float(vE.item())); vI_buf.append(float(vI.item()))
            next_vE_buf.append(next_vE); next_vI_buf.append(next_vI)
            done_buf.append(done)
            term_buf.append(bool(term))
            De_states.append(float(tv.item()))
            pA_buf.append(pA.squeeze(0).cpu().numpy())
            pB_buf.append(pB.squeeze(0).cpu().numpy())
            pmix_buf.append(pmix.squeeze(0).cpu().numpy())
            recent_int.append(r_int)
            ep_ext += float(r_ext); ep_int += float(r_int); ep_len += 1
            global_step += 1
            obs = nobs
            if done:
                success = 1 if term else 0
                recent_ext.append(ep_ext)
                ep_w.writerow(dict(episode_idx=ep_counter["deploy"], end_step=global_step,
                                   length=ep_len, ext_return=ep_ext, int_return=ep_int,
                                   terminated=success, source="deploy", beta=beta))
                ep_counter["deploy"] += 1
                ep_ext = 0.0; ep_int = 0.0
                recent_succ.append(success)
                if success:
                    recent_steps.append(ep_len)
                ep_len = 0
                obs, info = env.reset()
                beta = beta_pending

        vE_arr = np.asarray(vE_buf, dtype=np.float32)
        vI_arr = np.asarray(vI_buf, dtype=np.float32)
        act_arr = np.asarray(act_buf)

        def gae_stream(rews, v, next_v):
            T = len(rews); adv = np.zeros(T, dtype=np.float32); g = 0.0
            for t in reversed(range(T)):
                nv = next_v[t]
                bootstrap_nonterm = 0.0 if term_buf[t] else 1.0
                carry_nonterm = 0.0 if done_buf[t] else 1.0
                delta = rews[t] + gamma * nv * bootstrap_nonterm - v[t]
                g = delta + gamma * gae_lambda * carry_nonterm * g
                adv[t] = g
            return adv, adv + v


        advE_raw, retE = gae_stream(rew_ext_buf, vE_arr, next_vE_buf)
        advI_raw, retI = gae_stream(rew_int_buf, vI_arr, next_vI_buf)

        _retE = np.asarray(retE, dtype=np.float64); _rE = np.asarray(rew_ext_buf, dtype=np.float64)
        diag_retE_max = float(np.abs(_retE).max()); diag_retE_mean = float(np.abs(_retE).mean())
        diag_rE_max = float(np.abs(_rE).max())

        _tgt = np.asarray(retE); _var_tgt = float(_tgt.var())
        ev_vE = float(1.0 - ((_tgt - vE_arr).var() / _var_tgt)) if _var_tgt > 1e-8 else float("nan")

        def _spearman(x, y):
            n = len(x)
            if n < 3:
                return float("nan")
            rx = np.argsort(np.argsort(x)).astype(np.float64)
            ry = np.argsort(np.argsort(y)).astype(np.float64)
            rx -= rx.mean(); ry -= ry.mean()
            denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
            return float((rx * ry).sum() / denom) if denom > 1e-12 else float("nan")
        rankcorr_vE = _spearman(vE_arr, _tgt) if _var_tgt > 1e-8 else float("nan")

        sE = advE_raw.std() + 1e-8
        sI = advI_raw.std() + 1e-8
        advE_samp = advE_raw / sE
        advI_samp = advI_raw / sI
        advE_n = advE_samp
        advS_n = advE_samp + lam * advI_samp

        ob = torch.as_tensor(np.asarray(obs_buf), device=device)
        ac = torch.as_tensor(act_arr, device=device)
        logp_old = torch.as_tensor(np.asarray(logp_mix_buf), device=device)

        pA_sa_roll = np.asarray(pA_buf)[np.arange(rollout), act_arr]
        pB_sa_roll = np.asarray(pB_buf)[np.arange(rollout), act_arr]
        pmix_sa_roll = np.asarray(pmix_buf)[np.arange(rollout), act_arr]
        wA_roll = (1.0 - beta) * pA_sa_roll / (pmix_sa_roll + 1e-12)
        wB_roll = beta * pB_sa_roll / (pmix_sa_roll + 1e-12)
        tolA = np.minimum(wA_roll * delta_c, delta_phi)
        tolB = np.minimum(wB_roll * delta_c, delta_phi)
        tolA_t = torch.as_tensor(tolA.astype(np.float32), device=device)
        tolB_t = torch.as_tensor(tolB.astype(np.float32), device=device)

        tighten_frac = float(((tolA < delta_phi - 1e-9) | (tolB < delta_phi - 1e-9)).mean())
        advE_t = torch.as_tensor(advE_n, device=device)
        advS_t = torch.as_tensor(advS_n, device=device)
        retE_t = torch.as_tensor(retE, device=device)
        retI_t = torch.as_tensor(retI, device=device)
        idx = np.arange(rollout)
        actor_gradnorm_acc = []
        qlossE_first = None; qlossE_last = None
        klA_acc = klB_acc = cfA_acc = cfB_acc = vlA_acc = vlB_acc = 0.0
        n_upd = 0

        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, rollout, minibatch):
                mb = idx[s:s + minibatch]
                obm, acm = ob[mb], ac[mb]
                lA, vE_pred = actor.forward(obm)
                with torch.no_grad():
                    lB_det = advisor.logits(obm)
                logp_mix_A, _ = mixture_log_prob(lA, lB_det, beta, acm)
                rphi_A = torch.exp(logp_mix_A - logp_old[mb])
                tol_mb = tolA_t[mb]
                pg1 = rphi_A * advE_t[mb]
                pg2 = torch.clamp(rphi_A, 1.0 - tol_mb, 1.0 + tol_mb) * advE_t[mb]
                pol_A = -torch.min(pg1, pg2).mean()
                vlossE = ((vE_pred - retE_t[mb]) ** 2).mean()
                if qlossE_first is None:
                    qlossE_first = float(vlossE.item())
                qlossE_last = float(vlossE.item())
                distA = torch.distributions.Categorical(logits=lA)
                lossA = pol_A + vf_coef * vlossE - ent_coef_A * distA.entropy().mean()
                opt_A.zero_grad(); lossA.backward()
                _gnA = 0.0
                for _p in actor.parameters():
                    if _p.grad is not None:
                        _gnA += float(_p.grad.detach().pow(2).sum().item())
                actor_gradnorm_acc.append(_gnA ** 0.5)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt_A.step()
                with torch.no_grad():
                    klA_acc += float((logp_old[mb] - logp_mix_A.detach()).mean().item())
                    cfA_acc += float(((rphi_A.detach() - 1.0).abs() > tol_mb).float().mean().item())
                    vlA_acc += float(vlossE.item())

                lB, vI_pred = advisor.forward(obm)
                with torch.no_grad():
                    lA_det = actor.logits(obm)
                logp_mix_B, _ = mixture_log_prob(lA_det, lB, beta, acm)
                rphi_B = torch.exp(logp_mix_B - logp_old[mb])
                tol_mbB = tolB_t[mb]
                pg1 = rphi_B * advS_t[mb]
                pg2 = torch.clamp(rphi_B, 1.0 - tol_mbB, 1.0 + tol_mbB) * advS_t[mb]
                pol_B = -torch.min(pg1, pg2).mean()
                vlossI = ((vI_pred - retI_t[mb]) ** 2).mean()
                distB = torch.distributions.Categorical(logits=lB)
                lossB = pol_B + vf_coef * vlossI - ent_coef_B * distB.entropy().mean()
                opt_B.zero_grad(); lossB.backward()
                torch.nn.utils.clip_grad_norm_(advisor.parameters(), max_grad_norm)
                opt_B.step()
                with torch.no_grad():
                    klB_acc += float((logp_old[mb] - logp_mix_B.detach()).mean().item())
                    cfB_acc += float(((rphi_B.detach() - 1.0).abs() > tol_mbB).float().mean().item())
                    vlB_acc += float(vlossI.item())
                    n_upd += 1

                rnd_opt.zero_grad(); rnd.distill_loss(obm).backward(); rnd_opt.step()


        with torch.no_grad():
            _, vE_post = actor.forward(ob)
            vE_post_np = vE_post.cpu().numpy()
        rankcorr_qE_post = _spearman(vE_post_np, _tgt) if _var_tgt > 1e-8 else float("nan")
        qlossE_drop = (qlossE_first - qlossE_last) if (qlossE_first is not None) else float("nan")

        _nu = max(n_upd, 1)
        approx_kl_A = klA_acc / _nu; approx_kl_B = klB_acc / _nu
        clip_frac_A = cfA_acc / _nu; clip_frac_B = cfB_acc / _nu
        v_loss_A = vlA_acc / _nu; v_loss_B = vlB_acc / _nu

        with torch.no_grad():
            lA_post_full = actor.logits(ob)
            pA_post = torch.softmax(lA_post_full, dim=-1).cpu().numpy()
            lB_post_full = advisor.logits(ob)
            pB_post = torch.softmax(lB_post_full, dim=-1).cpu().numpy()
        ix_full = np.arange(rollout)
        pA_post_sa = pA_post[ix_full, act_arr]
        pB_post_sa = pB_post[ix_full, act_arr]
        pmix_post_sa = (1.0 - beta) * pA_post_sa + beta * pB_post_sa
        pA_sa_r = np.asarray(pA_buf)[ix_full, act_arr]
        pmix_sa_r = np.asarray(pmix_buf)[ix_full, act_arr]
        rA_post = pA_post_sa / (pA_sa_r + 1e-12)
        rphi_post = pmix_post_sa / (pmix_sa_r + 1e-12)
        r_Aphi = wA_roll / (1.0 - beta + 1e-12) * rA_post
        aE = advE_samp
        mt2 = float(np.mean((rphi_post * aE) ** 2))
        ct2 = float(np.mean((r_Aphi * aE) ** 2))
        r_Aphi_max = float(np.max(np.abs(r_Aphi)))
        frac_near0 = float(np.mean(pmix_sa_r < 1e-3))

        D_e = float(np.mean(De_states)) if De_states else 0.0
        I_e = beta * D_e
        D_beta = D_e

        pA_full = np.asarray(pA_buf); pB_full = np.asarray(pB_buf); pmix_full = np.asarray(pmix_buf)
        ent_A = float(-(pA_full.clip(1e-12) * np.log(pA_full.clip(1e-12))).sum(1).mean())
        ent_B = float(-(pB_full.clip(1e-12) * np.log(pB_full.clip(1e-12))).sum(1).mean())
        act_arr_full = np.asarray(act_buf); ix = np.arange(len(act_arr_full))
        pA_sa = pA_full[ix, act_arr_full]; pB_sa = pB_full[ix, act_arr_full]; pmix_sa = pmix_full[ix, act_arr_full]
        wA_mean = float(((1 - beta) * pA_sa / (pmix_sa + 1e-12)).mean())
        wB_mean = float((beta * pB_sa / (pmix_sa + 1e-12)).mean())

        isw_beta = (pA_sa - pB_sa) / (pmix_sa + 1e-12)
        G_beta = float((isw_beta * advE_samp).mean())

        if len(advE_samp) > 2:
            _mE = advE_samp.mean(); _mI = advI_samp.mean()
            H_lambda = float(((advI_samp - _mI) * (advE_samp - _mE)).mean())
        else:
            H_lambda = float("nan")

        beta_pending_pre = beta_pending
        withdraw_fired = 0; dbeta_raw = 0.0; dbeta_applied = 0.0
        if controller and G_beta > 0.0 and beta_pending > 0.0:
            if D_beta > d_beta_floor:
                dbeta = (1.0 - gamma) / gamma * delta_beta / D_beta
            else:
                dbeta = beta_pending
            dbeta_raw = float(dbeta)
            new_pending = max(0.0, beta_pending - dbeta)
            dbeta_applied = float(beta_pending - new_pending)
            beta_pending = new_pending
            withdraw_fired = 1
        beta_pending_post = beta_pending

        do_eval = (global_step % eval_every < rollout)
        if do_eval:
            ev_s, ev_st, ev_docc, ev_sc, ev_eps = actor_eval(actor, mk, device, n_episodes=20,
                                                             max_steps=max_steps, seed=20_000 + seed)
            last_eval = dict(success=ev_s, steps=ev_st,
                             occ_entropy=ev_sc["entropy"], goal_cell_occ=ev_sc["goal_cell_occ"],
                             goal_region_occ=ev_sc["goal_region_occ"])
            for _er in ev_eps:
                ep_w.writerow(dict(episode_idx=ep_counter["actor_eval"], end_step=global_step,
                                   length=_er["length"], ext_return=_er["ext_return"],
                                   int_return=float("nan"), terminated=_er["terminated"],
                                   source="actor_eval", beta=0.0))
                ep_counter["actor_eval"] += 1
            actor_snap_maps.append(ev_docc.astype(np.float32)); actor_snap_steps.append(global_step)
            actor_snap_entropy.append(ev_sc["entropy"]); actor_snap_goal_cell.append(ev_sc["goal_cell_occ"])
            actor_snap_goal_region.append(ev_sc["goal_region_occ"])

        mix_win_d, mix_win_v, mix_win_sc = env.snapshot_and_reset()
        mix_snap_maps.append(mix_win_d.astype(np.float32)); mix_snap_visits.append(mix_win_v.astype(np.int32))
        mix_snap_steps.append(global_step)
        mix_snap_entropy.append(mix_win_sc["entropy"]); mix_snap_goal_cell.append(mix_win_sc["goal_cell_occ"])
        mix_snap_goal_region.append(mix_win_sc["goal_region_occ"])

        row = dict(step=global_step,
                   mix_ext_return=float(np.mean(recent_ext[-20:])) if recent_ext else 0.0,
                   mix_success=float(np.mean(recent_succ[-20:])) if recent_succ else 0.0,
                   mix_steps=float(np.mean(recent_steps[-20:])) if recent_steps else float("nan"),
                   actor_eval_success=last_eval["success"],
                   actor_eval_steps=last_eval["steps"],
                   beta=beta, beta_pending=beta_pending, G_beta=G_beta, H_lambda=H_lambda,
                   ev_vE=ev_vE, rankcorr_vE=rankcorr_vE,
                   rankcorr_qE_post=rankcorr_qE_post, qlossE_first=qlossE_first, qlossE_drop=qlossE_drop,
                   D_e=D_e, I_e=I_e,
                   int_R=float(np.mean(recent_int[-1000:])) if recent_int else float("nan"),
                   ent_A=ent_A, ent_B=ent_B, wA_mean=wA_mean, wB_mean=wB_mean,
                   actor_gradnorm=float(np.mean(actor_gradnorm_acc)) if actor_gradnorm_acc else float("nan"),
                   withdraw_fired=withdraw_fired, dbeta_raw=dbeta_raw, dbeta_applied=dbeta_applied,
                   beta_pending_pre=beta_pending_pre, beta_pending_post=beta_pending_post,
                   approx_kl_A=approx_kl_A, approx_kl_B=approx_kl_B,
                   clip_frac_A=clip_frac_A, clip_frac_B=clip_frac_B,
                   v_loss_A=v_loss_A, v_loss_B=v_loss_B, tighten_frac=tighten_frac,
                   mt2=mt2, ct2=ct2, r_Aphi_max=r_Aphi_max, frac_near0=frac_near0,
                   deployed_occ_entropy=mix_win_sc["entropy"],
                   deployed_goal_cell_occ=mix_win_sc["goal_cell_occ"],
                   deployed_goal_region_occ=mix_win_sc["goal_region_occ"],
                   actor_occ_entropy=last_eval["occ_entropy"],
                   actor_goal_cell_occ=last_eval["goal_cell_occ"],
                   actor_goal_region_occ=last_eval["goal_region_occ"],
                   wall_s=time.time() - t0)
        csv_w.writerow(row); csv_f.flush()
        if global_step % log_every < rollout:
            print(f"  step {global_step:6d} | mix_succ {row['mix_success']:.2f} | "
                  f"act_eval {row['actor_eval_success']:.2f} | D_e {D_e:.3f} | "
                  f"entA {ent_A:.2f} entB {ent_B:.2f} | wA {wA_mean:.2f} wB {wB_mean:.2f} | "
                  f"gA {row['actor_gradnorm']:.1e} | rk {rankcorr_vE:+.2f}->{rankcorr_qE_post:+.2f} | vLE {qlossE_first:.3f}d{qlossE_drop:+.3f} | tighten {tighten_frac:.2f} | beta {beta:.3f} | {row['wall_s']:.0f}s")
            print(f"           critic | G_beta {G_beta:+.4f} | retE_max {diag_retE_max:.3f} mean {diag_retE_mean:.3f} | "
                  f"raw rE_max {diag_rE_max:.3f} | ev_vE {row['ev_vE']:.2f}")

    dt = time.time() - t0
    csv_f.close()
    ev_s, ev_st, actor_docc, ev_sc, ev_eps = actor_eval(actor, mk, device, n_episodes=40, max_steps=max_steps, seed=30_000 + seed)
    for _er in ev_eps:
        ep_w.writerow(dict(episode_idx=ep_counter["actor_eval"], end_step=global_step,
                           length=_er["length"], ext_return=_er["ext_return"],
                           int_return=float("nan"), terminated=_er["terminated"],
                           source="actor_eval", beta=0.0))
        ep_counter["actor_eval"] += 1
    ep_f.close()
    actor_snap_maps.append(actor_docc.astype(np.float32)); actor_snap_steps.append(global_step)
    actor_snap_entropy.append(ev_sc["entropy"]); actor_snap_goal_cell.append(ev_sc["goal_cell_occ"])
    actor_snap_goal_region.append(ev_sc["goal_region_occ"])
    mix_visit, mix_docc = env.occupancy_maps()
    np.savez(occ_path, actor_disc_occ=actor_docc, mix_disc_occ=mix_docc, mix_visit_counts=mix_visit,
             arm="bb", beta=beta, lam=lam, seed=seed,
             actor_eval_success=ev_s, actor_eval_steps=ev_st,
             deployed_occ_snapshots=np.asarray(mix_snap_maps, dtype=np.float32),
             deployed_occ_snapshot_visits=np.asarray(mix_snap_visits, dtype=np.int32),
             deployed_occ_snapshot_steps=np.asarray(mix_snap_steps, dtype=np.int64),
             deployed_snap_entropy=np.asarray(mix_snap_entropy, dtype=np.float32),
             deployed_snap_goal_cell_occ=np.asarray(mix_snap_goal_cell, dtype=np.float32),
             deployed_snap_goal_region_occ=np.asarray(mix_snap_goal_region, dtype=np.float32),
             actor_occ_snapshots=np.asarray(actor_snap_maps, dtype=np.float32),
             actor_occ_snapshot_steps=np.asarray(actor_snap_steps, dtype=np.int64),
             actor_snap_entropy=np.asarray(actor_snap_entropy, dtype=np.float32),
             actor_snap_goal_cell_occ=np.asarray(actor_snap_goal_cell, dtype=np.float32),
             actor_snap_goal_region_occ=np.asarray(actor_snap_goal_region, dtype=np.float32))
    config["n_actions"] = int(n_act)
    write_meta(meta_path, config, dt, extra=dict(ms_per_step=1000 * dt / total_steps,
                                                 occ_file=os.path.basename(occ_path),
                                                 final_actor_eval_success=ev_s))
    print(f"\nBB run done | size={size} steps={total_steps} beta={beta} lam={lam} seed={seed} | "
          f"{dt:.1f}s ({1000*dt/total_steps:.2f} ms/step) [TIMING]")
    print(f"  final actor-solo success={ev_s:.2f} steps={ev_st:.0f}")
    print(f"  curve -> {csv_path}\n  eps   -> {ep_path}\n  meta  -> {meta_path}\n  occ   -> {occ_path}")
    return dict(csv=csv_path, meta=meta_path, occ=occ_path, wall_s=dt)


if __name__ == "__main__":
    SIZE = 13
    STEPS = 100000
    MAX_STEPS = 1024
    SEEDS = [0, 1, 2, 3, 4]
    ROLLOUT = 512
    EPOCHS = 4
    MINIBATCH = 128
    LR = 2.5e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.975
    CLIP = 0.2
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    LR_ANNEAL_FLOOR = 0.1
    DELTA_C = 0.2
    DELTA_PHI = 0.1
    BETA = 0.45
    LAMBDAS = [0.05]
    CONTROLLER = False
    DELTA_BETA = 0.1
    D_BETA_FLOOR = 1e-3
    ENT_COEF = 0.01
    ENT_COEF_END = 0.01
    ENT_COEF_B = 0.1
    RND_LR = 1e-4
    EVAL_EVERY = 2500
    DEVICE = "cuda"
    RUNS_DIR = "runs"
    TAG = "bb_calib_held0p45_"
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
    ap.add_argument("--delta_c", type=float, default=DELTA_C)
    ap.add_argument("--delta_phi", type=float, default=DELTA_PHI)
    ap.add_argument("--vf_coef", type=float, default=VF_COEF)
    ap.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    ap.add_argument("--lr_anneal_floor", type=float, default=LR_ANNEAL_FLOOR)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--lam", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--controller", type=lambda s: str(s).lower() != "false", default=CONTROLLER)
    ap.add_argument("--delta_beta", type=float, default=DELTA_BETA)
    ap.add_argument("--d_beta_floor", type=float, default=D_BETA_FLOOR)
    ap.add_argument("--ent_coef", type=float, default=ENT_COEF)
    ap.add_argument("--ent_coef_end", type=float, default=ENT_COEF_END)
    ap.add_argument("--ent_coef_B", type=float, default=ENT_COEF_B)
    ap.add_argument("--rnd_lr", type=float, default=RND_LR)
    ap.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--runs_dir", default=RUNS_DIR)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [device] cuda requested but unavailable -> falling back to cpu")
        args.device = "cpu"
    L = "L2 (controller)" if args.controller else "L1 (fixed beta)"
    print(f"BB {L}: beta={args.beta} lam={args.lam} | lr={args.lr} anneal_floor={args.lr_anneal_floor} "
          f"gae_lam={args.gae_lambda} dc={args.delta_c} dphi={args.delta_phi} vf={args.vf_coef} gnorm={args.max_grad_norm} "
          f"ent={args.ent_coef}->{args.ent_coef_end} ent_B={args.ent_coef_B} | steps={args.steps} "
          f"max_steps={args.max_steps} | device={args.device} | tag='{args.tag}'")
    for lam in args.lam:
        for seed in args.seed:
            print(f"\n=== BB {L} | beta={args.beta} lam={lam} seed={seed} ===")
            train_bb(size=args.size, total_steps=args.steps, max_steps=args.max_steps,
                     rollout=args.rollout, epochs=args.epochs, minibatch=args.minibatch,
                     lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda, clip=args.clip,
                     ent_coef=args.ent_coef, ent_coef_end=args.ent_coef_end, ent_coef_B=args.ent_coef_B,
                     vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                     delta_c=args.delta_c, delta_phi=args.delta_phi,
                     lr_anneal_floor=args.lr_anneal_floor, beta=args.beta, lam=lam, rnd_lr=args.rnd_lr,
                     controller=args.controller, delta_beta=args.delta_beta, d_beta_floor=args.d_beta_floor,
                     eval_every=args.eval_every, seed=seed, device=args.device,
                     runs_dir=args.runs_dir, tag=args.tag)