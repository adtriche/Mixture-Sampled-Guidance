import argparse
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.envs import FourRoomsEnv
from minigrid.wrappers import RGBImgPartialObsWrapper
import torch
import torch.nn as nn


class FourRoomsTrapEnv(FourRoomsEnv):
    def __init__(self, size=11, agent_pos=(1, 1), goal_pos=None, max_steps=200, **kwargs):
        self._size = size
        if goal_pos is None:
            goal_pos = (size - 2, size - 2)
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos, max_steps=max_steps, **kwargs)
        self.size = size
        self.width = size
        self.height = size
        self.observation_space = self._build_obs_space()

    def _build_obs_space(self):
        from gymnasium import spaces as _sp
        import numpy as _np
        image_space = _sp.Box(low=0, high=255,
                              shape=(self.agent_view_size, self.agent_view_size, 3), dtype=_np.uint8)
        return _sp.Dict({"image": image_space, "direction": _sp.Discrete(4),
                         "mission": self.observation_space["mission"]})


def make_fourrooms_trap(size=11, agent_pos=(1, 1), goal_pos=None, trap_cells=None,
                        max_steps=200, noise_seed=0, tile_size=8, gamma=0.99):
    if goal_pos is None:
        goal_pos = (size - 2, size - 2)
    base = FourRoomsTrapEnv(size=size, agent_pos=agent_pos, goal_pos=goal_pos,
                            max_steps=max_steps, render_mode="rgb_array")
    rgb = RGBImgPartialObsWrapper(base, tile_size=tile_size)
    if trap_cells is None:
        gx, gy = goal_pos
        trap_cells = [(gx - 3, gy), (gx - 2, gy), (gx - 3, gy - 1), (gx - 2, gy - 1)]
    return NoisyTVTrapWrapper(rgb, trap_cells=trap_cells, noise_seed=noise_seed,
                              tile_size=tile_size, gamma=gamma)


class NoisyTVTrapWrapper(gym.Wrapper):
    def __init__(self, env, trap_cells, noise_seed=0, tile_size=8, gamma=0.99):
        super().__init__(env)
        self.trap_cells = set(map(tuple, trap_cells))
        self.tile_size = tile_size
        self.gamma = gamma
        self._noise_rng = np.random.default_rng(noise_seed)
        self._mg = env.unwrapped
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t = 0
        self._disc = 1.0
        self.trap_occ = 0.0
        self.goal_occ = 0.0
        self._goal_cell = self._find_goal()
        obs = self._inject(obs)
        info = self._augment_info(info)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        ax, ay = tuple(self._mg.agent_pos)
        if (ax, ay) in self.trap_cells:
            self.trap_occ += self._disc
        if self._goal_cell is not None and (ax, ay) == self._goal_cell:
            self.goal_occ += self._disc
        self._disc *= self.gamma
        self._t += 1
        obs = self._inject(obs)
        info = self._augment_info(info)
        return obs, rew, term, trunc, info

    def _inject(self, obs):
        img = obs["image"]
        self._last_trap_visible = False
        self._last_trap_in_frustum = False
        grid, vis_mask = self._mg.gen_obs_grid()
        view = self._mg.agent_view_size
        ts = self.tile_size
        for (tx, ty) in self.trap_cells:
            vc = self._world_to_view(tx, ty)
            if vc is None:
                continue
            vx, vy = vc
            if not (0 <= vx < view and 0 <= vy < view):
                continue
            self._last_trap_in_frustum = True
            if not vis_mask[vx, vy]:
                continue
            self._last_trap_visible = True
            r0, c0 = vy * ts, vx * ts
            noise = self._noise_rng.integers(0, 256, size=(ts, ts, 3), dtype=np.uint8)
            img[r0:r0 + ts, c0:c0 + ts, :] = noise
        obs["image"] = img
        return obs

    def _world_to_view(self, x, y):
        rc = self._mg.relative_coords(x, y)
        return rc

    def _augment_info(self, info):
        info = dict(info)
        info["trap_visible"] = bool(getattr(self, "_last_trap_visible", False))
        info["trap_in_frustum"] = bool(getattr(self, "_last_trap_in_frustum", False))
        info["trap_occ"] = self.trap_occ
        info["goal_occ"] = self.goal_occ
        info["agent_pos"] = tuple(self._mg.agent_pos)
        return info

    def _find_goal(self):
        for i in range(self._mg.width):
            for j in range(self._mg.height):
                c = self._mg.grid.get(i, j)
                if c is not None and c.type == "goal":
                    return (i, j)
        return None

def _to_chw_float(x):
    x = x.float() / 255.0
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    return x


class LeanEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.feat_dim = feat_dim
        self._head = None

    def forward(self, x):
        h = self.conv(_to_chw_float(x))
        if self._head is None:
            self._head = nn.Linear(h.shape[1], self.feat_dim).to(h.device)
        return self._head(h)


class DetailEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.feat_dim = feat_dim
        self._head = None

    def forward(self, x):
        x = _to_chw_float(x)
        x = (x - x.mean()) / (x.std() + 1e-8)
        h = self.conv(x)
        if self._head is None:
            self._head = nn.Linear(h.shape[1], self.feat_dim).to(h.device)
        return self._head(h)


class RNDPair(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.target = DetailEncoder(feat_dim=feat_dim)
        self.predictor = DetailEncoder(feat_dim=feat_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def error(self, obs):
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        return ((p - t) ** 2).mean(dim=1)


def trap_persistence_check(size=11, steps=4000, lr=1e-4, device="cpu", noise_seed=0, seed=0):
    env = make_fourrooms_trap(size=size, noise_seed=noise_seed)
    rnd = RNDPair().to(device)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)
    buckets = {"visible": [], "occluded": [], "absent": []}
    window = []
    t0 = time.time()
    for t in range(steps):
        a = int(rng.integers(0, env.action_space.n))
        obs, rew, term, trunc, info = env.step(a)
        o = torch.as_tensor(obs["image"][None], device=device)
        err = rnd.error(o)
        opt.zero_grad(); err.mean().backward(); opt.step()
        e = float(err.item())
        if info["trap_visible"]:
            bucket = "visible"
        elif info["trap_in_frustum"]:
            bucket = "occluded"
        else:
            bucket = "absent"
        buckets[bucket].append((t, e))
        window.append((t, bucket, e))
        if term or trunc:
            obs, info = env.reset()
    dt = time.time() - t0

    def tail_mean(pairs, frac=0.5):
        if not pairs:
            return float("nan")
        tail = pairs[int(len(pairs) * (1 - frac)):]
        return float(np.mean([e for _, e in tail]))

    def head_mean(pairs, frac=0.25):
        if not pairs:
            return float("nan")
        head = pairs[:max(1, int(len(pairs) * frac))]
        return float(np.mean([e for _, e in head]))

    print(f"\ntrap-persistence check | size={size} steps={steps} | {dt:.1f}s "
          f"({1000*dt/steps:.2f} ms/step) [TIMING for budget est.]")
    print(f"  visited counts: visible={len(buckets['visible'])} "
          f"occluded={len(buckets['occluded'])} absent={len(buckets['absent'])}")
    for b in ("visible", "occluded", "absent"):
        print(f"  err[{b:8s}] head(first 25%)={head_mean(buckets[b]):.4e}  "
              f"tail(last 50%)={tail_mean(buckets[b]):.4e}")
    vis_tail = tail_mean(buckets["visible"]); abs_tail = tail_mean(buckets["absent"])
    print(f"  GAP tail: err[visible]/err[absent] = "
          f"{vis_tail/abs_tail:.2f}x" if abs_tail and not np.isnan(abs_tail) else "  GAP: n/a")
    print("  GREEN if err[visible] tail stays high (>= head) AND err[absent] tail falls below it"
          " AND the gap is large/sustained.")
    return buckets, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    trap_persistence_check(size=args.size, steps=args.steps, device=args.device, seed=args.seed)