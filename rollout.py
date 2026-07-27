from dataclasses import dataclass
import numpy as np


@dataclass
class Rollouts:
    s: np.ndarray
    a: np.ndarray
    s_next: np.ndarray
    r_ext: np.ndarray
    r_int: np.ndarray
    term: np.ndarray
    trunc: np.ndarray
    ep_id: np.ndarray
    n_episodes: int

    @property
    def done(self):
        return self.term | self.trunc

    def reward(self, stream):
        if stream == "ext":
            return self.r_ext
        if stream == "int":
            return self.r_int
        raise ValueError(f"unknown stream {stream!r}")


def collect_rollouts(env, policy, n_episodes, *, seed=0):
    policy = np.asarray(policy, dtype=float)
    rng = np.random.default_rng(seed)
    s_, a_, sn_, re_, ri_, tm_, tr_, ep_ = ([] for _ in range(8))
    for ep in range(n_episodes):
        s = env.reset(seed=int(rng.integers(1 << 31)))
        while True:
            p = policy[s]
            p = p / p.sum()
            a = int(rng.choice(p.shape[0], p=p))
            ns, re, ri, terminated, truncated, _ = env.step(a)
            s_.append(s); a_.append(a); sn_.append(ns)
            re_.append(re); ri_.append(ri)
            tm_.append(bool(terminated)); tr_.append(bool(truncated)); ep_.append(ep)
            s = ns
            if terminated or truncated:
                break
    return Rollouts(
        s=np.array(s_, dtype=int), a=np.array(a_, dtype=int), s_next=np.array(sn_, dtype=int),
        r_ext=np.array(re_, dtype=float), r_int=np.array(ri_, dtype=float),
        term=np.array(tm_, dtype=bool), trunc=np.array(tr_, dtype=bool),
        ep_id=np.array(ep_, dtype=int), n_episodes=n_episodes)


def empirical_occupancy(roll, n_states, gamma):
    d = np.zeros(n_states)
    for ep in range(roll.n_episodes):
        idx = np.flatnonzero(roll.ep_id == ep)
        if idx.size == 0:
            continue
        for t, i in enumerate(idx):
            d[roll.s[i]] += (1.0 - gamma) * gamma ** t
        last = idx[-1]
        if roll.term[last]:
            d[roll.s_next[last]] += gamma ** idx.size
    d /= roll.n_episodes
    return d


def compute_gae(roll, V, stream, gamma, gae_lambda):
    V = np.asarray(V, dtype=float)
    r = roll.reward(stream)
    delta = r + gamma * (~roll.term) * V[roll.s_next] - V[roll.s]
    gcont = gamma * gae_lambda * (~roll.done)
    N = r.shape[0]
    adv = np.empty(N)
    lastgae = 0.0
    for t in range(N - 1, -1, -1):
        lastgae = delta[t] + gcont[t] * lastgae
        adv[t] = lastgae
    value_target = adv + V[roll.s]
    return adv, value_target


def bucket_by_sa(roll, values, n_states, n_actions):
    total = np.zeros((n_states, n_actions))
    count = np.zeros((n_states, n_actions))
    np.add.at(total, (roll.s, roll.a), values)
    np.add.at(count, (roll.s, roll.a), 1.0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return mean, count


def bucket_by_s(roll, values, n_states):
    total = np.zeros(n_states)
    count = np.zeros(n_states)
    np.add.at(total, roll.s, values)
    np.add.at(count, roll.s, 1.0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return mean, count


def fit_value_tabular(roll, stream, gamma, n_states, *, gae_lambda=0.95, n_iters=50, V_init=None, damping=0.0):
    V = np.zeros(n_states) if V_init is None else np.array(V_init, dtype=float)
    for _ in range(n_iters):
        _, target = compute_gae(roll, V, stream, gamma, gae_lambda)
        mean, count = bucket_by_s(roll, target, n_states)
        V_new = np.where(count > 0, mean, V)
        if damping > 0.0:
            V_new = (1.0 - damping) * V_new + damping * V
        V = V_new
    return V