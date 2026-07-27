from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple
import numpy as np

Cell = Tuple[int, int]

class GridEnv:
    _MOVES: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
    N_ACTIONS: int = 4

    def __init__(
        self,
        size: int,
        start: Cell,
        goal: Cell,
        distractors: Optional[Sequence[Tuple[Cell, float]]] = None,
        r_goal: float = 1.0,
        *,
        k: float = 3.0,
        c: float = 8.0,
        gamma: Optional[float] = None,
        max_steps: Optional[int] = None,
        slip: float = 0.0,
        intrinsic_field: Optional["np.ndarray"] = None,
        wall_edges: Optional[Sequence[Tuple[Cell, Cell]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        if not 0.0 <= slip <= 1.0:
            raise ValueError("slip must be in [0, 1]")

        self.size = int(size)
        self.n_states = self.size * self.size
        self.n_actions = self.N_ACTIONS

        self.start = self._check_cell(start, "start")
        self.goal = self._check_cell(goal, "goal")
        if self.start == self.goal:
            raise ValueError("start and goal must differ")

        raw = list(distractors) if distractors is not None else []
        self.distractors: list[Tuple[Cell, float]] = []
        seen: set[Cell] = set()
        for cell, value in raw:
            cell = self._check_cell(cell, "distractor")
            if cell == self.goal:
                raise ValueError("a distractor cannot coincide with the goal")
            if cell == self.start:
                raise ValueError("a distractor cannot coincide with the start")
            if cell in seen:
                raise ValueError(f"duplicate distractor cell {cell}")
            seen.add(cell)
            self.distractors.append((cell, float(value)))

        self.r_goal = float(r_goal)
        self.slip = float(slip)
        self.k = float(k)
        self.c = float(c)

        self.d_sg = (
            abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])
        )
        self.gamma = (
            float(gamma) if gamma is not None else 1.0 - 1.0 / (self.k * self.d_sg)
        )
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1); got {self.gamma}")
        self.max_steps = (
            int(max_steps) if max_steps is not None else math.ceil(self.c * self.d_sg)
        )
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self._goal_s = self._to_index(self.goal)
        self._start_s = self._to_index(self.start)
        self._distractor_value = np.zeros(self.n_states, dtype=np.float64)
        for cell, value in self.distractors:
            self._distractor_value[self._to_index(cell)] = value

        if intrinsic_field is not None:
            field = np.asarray(intrinsic_field, dtype=np.float64).reshape(-1)
            if field.shape[0] != self.n_states:
                raise ValueError(
                    f"intrinsic_field must have {self.n_states} entries; got {field.shape[0]}"
                )
            if np.any(field < 0.0):
                raise ValueError("intrinsic_field values must be non-negative")
            if field[self._start_s] != 0.0:
                raise ValueError("intrinsic_field must be zero at the start state")
            self._distractor_value = field
            self.intrinsic_field = field
        else:
            self.intrinsic_field = None


        self.wall_edges: list[Tuple[Cell, Cell]] = []
        self._blocked: Optional[set] = None
        if wall_edges is not None:
            blocked: set = set()
            for cell_a, cell_b in wall_edges:
                a = self._check_cell(cell_a, "wall")
                b = self._check_cell(cell_b, "wall")
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                    raise ValueError(
                        f"wall edge {cell_a}-{cell_b} must be orthogonally adjacent"
                    )
                sa, sb = self._to_index(a), self._to_index(b)
                blocked.add((sa, sb))
                blocked.add((sb, sa))
                self.wall_edges.append((a, b))
            self._blocked = blocked

        self._rng = np.random.default_rng(seed)
        (
            self._P,
            self._R_ext,
            self._R_int,
            self._terminal_mask,
            self._mu0,
        ) = self._build_model()

        # sampling state
        self._s: Optional[int] = None
        self._t: int = 0

    def _check_cell(self, cell: Cell, name: str) -> Cell:
        r, c = int(cell[0]), int(cell[1])
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise ValueError(f"{name} {cell} out of bounds for size {self.size}")
        return (r, c)

    def _to_index(self, cell: Cell) -> int:
        return cell[0] * self.size + cell[1]

    def _to_cell(self, s: int) -> Cell:
        return (s // self.size, s % self.size)

    def _move(self, s: int, a: int) -> int:
        r, c = self._to_cell(s)
        dr, dc = self._MOVES[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return s
        ns = nr * self.size + nc
        if self._blocked is not None and (s, ns) in self._blocked:
            return s
        return ns

    def _build_model(self):
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S), dtype=np.float64)
        R_ext = np.zeros((S, A), dtype=np.float64)
        R_int = np.zeros((S, A), dtype=np.float64)
        terminal_mask = np.zeros(S, dtype=bool)
        terminal_mask[self._goal_s] = True

        for s in range(S):
            if terminal_mask[s]:
                P[s, :, s] = 1.0
                continue
            for a in range(A):
                if self.slip == 0.0:
                    P[s, a, self._move(s, a)] += 1.0
                else:
                    P[s, a, self._move(s, a)] += 1.0 - self.slip
                    for a2 in range(A):
                        P[s, a, self._move(s, a2)] += self.slip / A
                R_ext[s, a] = self.r_goal * P[s, a, self._goal_s]
                R_int[s, a] = self._distractor_value[s]

        mu0 = np.zeros(S, dtype=np.float64)
        mu0[self._start_s] = 1.0
        return P, R_ext, R_int, terminal_mask, mu0

    @property
    def P(self) -> np.ndarray:
        """Transition kernel, shape (n_states, n_actions, n_states)."""
        return self._P

    @property
    def R_ext(self) -> np.ndarray:
        """Expected extrinsic reward, shape (n_states, n_actions)."""
        return self._R_ext

    @property
    def R_int(self) -> np.ndarray:
        """Expected intrinsic reward, shape (n_states, n_actions)."""
        return self._R_int

    @property
    def terminal_mask(self) -> np.ndarray:
        """Boolean mask of absorbing states, shape (n_states,)."""
        return self._terminal_mask

    @property
    def mu0(self) -> np.ndarray:
        """Initial-state distribution, shape (n_states,)."""
        return self._mu0


    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._s = self._start_s
        self._t = 0
        return self._s

    def step(self, action: int):
        if self._s is None:
            raise RuntimeError("call reset() before step()")
        if self._terminal_mask[self._s]:
            raise RuntimeError("episode already terminated; call reset()")

        s = self._s
        if self.slip > 0.0 and self._rng.random() < self.slip:
            a = int(self._rng.integers(self.n_actions))
        else:
            a = int(action)
        ns = self._move(s, a)

        r_int = float(self._distractor_value[s])
        r_ext = self.r_goal if ns == self._goal_s else 0.0

        self._s = ns
        self._t += 1
        terminated = bool(ns == self._goal_s)
        truncated = bool(self._t >= self.max_steps and not terminated)
        info = {"cell": self._to_cell(ns), "t": self._t}
        return ns, r_ext, r_int, terminated, truncated, info

    def __repr__(self) -> str:
        return (
            f"GridEnv(size={self.size}, start={self.start}, goal={self.goal}, "
            f"distractors={self.distractors}, r_goal={self.r_goal}, slip={self.slip}, "
            f"d_sg={self.d_sg}, gamma={self.gamma:.4f}, max_steps={self.max_steps}, "
            f"walls={len(self.wall_edges)})"
        )


def make_basin_field(
    size: int,
    center: Cell,
    peak: float,
    radius: float,
    shape: str = "linear",
) -> "np.ndarray":

    cr, cc = int(center[0]), int(center[1])
    field = np.zeros(size * size, dtype=np.float64)
    for s in range(size * size):
        r, c = s // size, s % size
        d = abs(r - cr) + abs(c - cc)
        if shape == "linear":
            field[s] = peak * max(0.0, 1.0 - d / radius)
        elif shape == "geometric":
            field[s] = peak * (radius ** d)
        else:
            raise ValueError(f"unknown basin shape {shape!r}")
    return field


def make_vwall(
    size: int,
    left_col: int,
    door_rows,
) -> list:
    if not (0 <= left_col < size - 1):
        raise ValueError(f"left_col must be in [0, {size - 2}]; got {left_col}")
    doors = {int(door_rows)} if isinstance(door_rows, int) else {int(r) for r in door_rows}
    edges = []
    for r in range(size):
        if r in doors:
            continue
        edges.append(((r, left_col), (r, left_col + 1)))
    return edges


def make_room_field(
    size: int,
    payout: float,
    col_min: int,
    col_max: Optional[int] = None,
    exclude: Sequence[Cell] = (),
) -> "np.ndarray":

    hi = size - 1 if col_max is None else int(col_max)
    field = np.zeros(size * size, dtype=np.float64)
    ex = {(int(r), int(c)) for (r, c) in exclude}
    for s in range(size * size):
        r, c = s // size, s % size
        if col_min <= c <= hi and (r, c) not in ex:
            field[s] = payout
    return field


if __name__ == "__main__":
    env = GridEnv(size=10, start=(0, 0), goal=(9, 9), distractors=[((2, 7), 0.5)])
    print(env)
    s = env.reset(seed=0)
    total_ext = 0.0
    for _ in range(env.max_steps):
        s, re, ri, term, trunc, _ = env.step(action=1)  # always "down"
        total_ext += re
        if term or trunc:
            break
    print(f"smoke rollout: ext={total_ext}, terminated={term}, truncated={trunc}")