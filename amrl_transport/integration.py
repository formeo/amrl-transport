"""
Transport-backed RL environment.

Drop-in replacement for SINGROUP's ``RealExpEnv`` from
``AMRL.Environment.Env_new``, but using the ``STMTransport`` interface
instead of hardcoded Createc COM calls.

Usage:
    from amrl_transport.transport import SimulatorTransport
    from amrl_transport.integration import TransportEnv

    stm = SimulatorTransport(seed=42)
    stm.connect()

    env = TransportEnv(
        transport=stm,
        step_nm=0.2,
        max_mvolt=20,
        max_pcurrent_to_mvolt_ratio=2850,
        goal_nm=2.0,
        template=my_template,
        current_jump=4,
        im_size_nm=10.0,
        offset_nm=np.array([0, 0]),
        manip_limit_nm=np.array([-4, 4, -4, 4]),
        pixel=128,
        template_max_y=60,
        scan_mV=100,
        max_len=20,
        load_weight='atom_move_detector.pth',
    )

    state, info = env.reset()
    next_state, reward, done, info = env.step(action)

The API is identical to RealExpEnv so existing RL training code
(sac_agent, notebooks) works without changes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .transport.protocol import STMTransport


class TransportEnv:
    """
    RL environment for atom manipulation via STMTransport.

    Compatible with the SINGROUP ``RealExpEnv`` interface — same
    ``reset()`` / ``step(action)`` contract, same state/reward semantics.

    The key difference: instead of ``Createc_Controller``, this uses
    any ``STMTransport`` instance (Createc, Nanonis, Simulator, ...).

    Parameters
    ----------
    transport : STMTransport
        An already-connected STM transport instance.
    step_nm : float
        RL step size in nm.
    max_mvolt : float
        Maximum manipulation bias in mV.
    max_pcurrent_to_mvolt_ratio : float
        Maximum current/voltage ratio for manipulation.
    goal_nm : float
        Maximum goal distance in nm.
    template : ndarray
        Template image for atom localization.
    current_jump : float
        Threshold for current jump detection.
    im_size_nm : float
        Scan image size in nm.
    offset_nm : ndarray
        XY offset in nm.
    manip_limit_nm : ndarray
        [x_min, x_max, y_min, y_max] manipulation limits.
    pixel : int
        Scan pixel count.
    template_max_y : int
        Template matching Y limit.
    scan_mV : float
        Scan bias in mV.
    max_len : int
        Maximum steps per episode.
    load_weight : str
        Path to AtomJumpDetector weights.
    pull_back_mV : float, optional
        Bias for pulling atom back to safe zone.
    pull_back_pA : float, optional
        Current for pulling atom back.
    random_scan_rate : float
        Rate of random scans for exploration.
    correct_drift : bool
        Whether to correct thermal drift.
    bottom : bool
        Template matching direction.
    atom_detector : callable, optional
        Custom atom detection function. If None, uses the original
        template-matching + blob detection from SINGROUP code.
    """

    def __init__(
        self,
        transport: STMTransport,
        step_nm: float,
        max_mvolt: float,
        max_pcurrent_to_mvolt_ratio: float,
        goal_nm: float,
        template: np.ndarray,
        current_jump: float,
        im_size_nm: float,
        offset_nm: np.ndarray,
        manip_limit_nm: np.ndarray,
        pixel: int,
        template_max_y: int,
        scan_mV: float,
        max_len: int,
        load_weight: str,
        pull_back_mV: Optional[float] = None,
        pull_back_pA: Optional[float] = None,
        random_scan_rate: float = 0.5,
        correct_drift: bool = False,
        bottom: bool = True,
        atom_detector=None,
    ):
        self.transport = transport
        self.step_nm = step_nm
        self.max_mvolt = max_mvolt
        self.max_pcurrent_to_mvolt_ratio = max_pcurrent_to_mvolt_ratio
        self.pixel = pixel
        self.goal_nm = goal_nm
        self.template = template
        self.current_jump = current_jump
        self.manip_limit_nm = manip_limit_nm
        self.inner_limit_nm = (
            manip_limit_nm + np.array([1, -1, 1, -1])
            if manip_limit_nm is not None
            else None
        )
        self.offset_nm = offset_nm
        self.len_nm = im_size_nm
        self.scan_mV = scan_mV

        self.default_reward = -1
        self.default_reward_done = 1
        self.max_len = max_len
        self.correct_drift = correct_drift
        self.template_max_y = template_max_y

        self.lattice_constant = 0.288
        self.precision_lim = self.lattice_constant * np.sqrt(3) / 3
        self.bottom = bottom

        self.pull_back_mV = pull_back_mV or 10
        self.pull_back_pA = pull_back_pA or 57000
        self.random_scan_rate = random_scan_rate

        # Atom jump detector — lazy import to avoid hard dependency
        self._atom_move_detector = None
        self._load_weight = load_weight
        self._init_detector()

        # Custom atom detector (override template matching)
        self._atom_detector = atom_detector

        # State
        self.atom_absolute_nm = None
        self.atom_relative_nm = None
        self.state = None
        self.len = 0
        self.accuracy, self.true_positive, self.true_negative = [], [], []

    def _init_detector(self):
        """Initialize the atom jump detector CNN."""
        try:
            from AMRL.Environment.atom_jump_detection import AtomJumpDetector_conv
            self._atom_move_detector = AtomJumpDetector_conv(
                data_len=2048, load_weight=self._load_weight
            )
        except ImportError:
            # Fallback: gradient-based detection only
            self._atom_move_detector = None

    # ── Core RL interface ─────────────────────────────────────

    def reset(self, update_conv_net: bool = True):
        """Reset the environment. Returns (state, info)."""
        self.len = 0

        # Update atom move detector if available
        if (
            self._atom_move_detector is not None
            and len(self._atom_move_detector.currents_val)
            > self._atom_move_detector.batch_size
            and update_conv_net
        ):
            acc, tp, tn = self._atom_move_detector.eval()
            self.accuracy.append(acc)
            self.true_positive.append(tp)
            self.true_negative.append(tn)
            self._atom_move_detector.train()

        if self.atom_absolute_nm is None or self.atom_relative_nm is None:
            self.atom_absolute_nm, self.atom_relative_nm = self._scan_atom()

        if self.out_of_range(self.atom_absolute_nm, self.inner_limit_nm):
            self._pull_atom_back()
            self.atom_absolute_nm, self.atom_relative_nm = self._scan_atom()

        goal_nm = self.lattice_constant + np.random.random() * (
            self.goal_nm - self.lattice_constant
        )
        self.atom_start_absolute_nm = self.atom_absolute_nm
        self.atom_start_relative_nm = self.atom_relative_nm

        dest_rel, dest_abs, self.goal = self._get_destination(
            self.atom_start_relative_nm, self.atom_start_absolute_nm, goal_nm
        )
        self.destination_relative_nm = dest_rel
        self.destination_absolute_nm = dest_abs

        self.state = np.concatenate((
            self.goal,
            (self.atom_absolute_nm - self.atom_start_absolute_nm) / self.goal_nm,
        ))
        self.dist_destination = goal_nm

        info = {
            "start_absolute_nm": self.atom_start_absolute_nm,
            "start_relative_nm": self.atom_start_relative_nm,
            "goal_absolute_nm": self.destination_absolute_nm,
            "goal_relative_nm": self.destination_relative_nm,
        }
        return self.state, info

    def step(self, action):
        """Take a step. Returns (next_state, reward, done, info)."""
        rets = self._action_to_manip_params(action)
        x_s, y_s, x_e, y_e, mvolt, pcurrent = rets

        current_series, d = self._execute_manipulation(
            x_s, y_s, x_e, y_e, mvolt, pcurrent
        )

        info = {
            "current_series": current_series,
            "d": d,
            "start_nm": np.array([x_s, y_s]),
            "end_nm": np.array([x_e, y_e]),
        }

        done = False
        self.len += 1
        done = self.len == self.max_len

        if not done and current_series is not None:
            jump = self._detect_current_jump(current_series)
        else:
            jump = False

        if done or jump:
            old_pos = self.atom_absolute_nm
            self.atom_absolute_nm, self.atom_relative_nm = self._scan_atom()
            self.dist_destination = np.linalg.norm(
                self.atom_absolute_nm - self.destination_absolute_nm
            )
            dist_start = np.linalg.norm(
                self.atom_absolute_nm - self.atom_start_absolute_nm
            )
            dist_last = np.linalg.norm(self.atom_absolute_nm - old_pos)

            oor = self.out_of_range(self.atom_absolute_nm, self.manip_limit_nm)
            in_precision = self.dist_destination < self.precision_lim
            too_far = dist_start > 1.5 * self.goal_nm
            done = done or too_far or in_precision or oor

            if self._atom_move_detector is not None:
                self._atom_move_detector.push(current_series, dist_last)

        next_state = np.concatenate((
            self.goal,
            (self.atom_absolute_nm - self.atom_start_absolute_nm) / self.goal_nm,
        ))

        reward = self._compute_reward(self.state, next_state)
        info["dist_destination"] = self.dist_destination
        info["atom_absolute_nm"] = self.atom_absolute_nm
        self.state = next_state

        return next_state, reward, done, info

    # ── Transport-backed operations ───────────────────────────

    def _scan_atom(self):
        """Scan and extract atom position using the transport."""
        result = self.transport.scan_image(
            size_nm=self.len_nm,
            offset_nm=self.offset_nm,
            pixel=self.pixel,
            bias_mv=self.scan_mV,
        )

        if self._atom_detector is not None:
            # Custom detector (e.g., ML-based)
            abs_nm, rel_nm = self._atom_detector(
                result.img_forward, result.img_backward,
                self.template, self.offset_nm, self.len_nm,
            )
        else:
            # Original SINGROUP detection
            try:
                from AMRL.Environment.get_atom_coordinate import (
                    get_atom_coordinate_nm,
                )
                abs_nm, rel_nm = get_atom_coordinate_nm(
                    result.img_forward,
                    result.img_backward,
                    self.offset_nm,
                    self.len_nm,
                    self.template,
                    self.template_max_y,
                    self.bottom,
                )
            except ImportError:
                # Fallback: find brightest pixel
                img = result.img_forward
                idx = np.unravel_index(np.argmax(img), img.shape)
                y_frac, x_frac = idx[0] / img.shape[0], idx[1] / img.shape[1]
                x_nm = self.offset_nm[0] - self.len_nm / 2 + x_frac * self.len_nm
                y_nm = self.offset_nm[1] - self.len_nm / 2 + y_frac * self.len_nm
                abs_nm = np.array([x_nm, y_nm])
                rel_nm = abs_nm - self.offset_nm

        return abs_nm, rel_nm

    def _execute_manipulation(self, x_s, y_s, x_e, y_e, mvolt, pcurrent):
        """Execute lateral manipulation via the transport."""
        x_s += self.atom_absolute_nm[0]
        x_e += self.atom_absolute_nm[0]
        y_s += self.atom_absolute_nm[1]
        y_e += self.atom_absolute_nm[1]

        x_kw = {"a_min": self.manip_limit_nm[0], "a_max": self.manip_limit_nm[1]}
        y_kw = {"a_min": self.manip_limit_nm[2], "a_max": self.manip_limit_nm[3]}

        x_s = np.clip(x_s, **x_kw)
        y_s = np.clip(y_s, **y_kw)
        x_e = np.clip(x_e, **x_kw)
        y_e = np.clip(y_e, **y_kw)

        if [x_s, y_s] == [x_e, y_e]:
            return None, None

        data = self.transport.lateral_manipulation(
            x_start_nm=x_s,
            y_start_nm=y_s,
            x_end_nm=x_e,
            y_end_nm=y_e,
            bias_mv=mvolt,
            current_pa=pcurrent,
            offset_nm=self.offset_nm,
            size_nm=self.len_nm,
        )

        if data is not None:
            current = np.array(data.current).flatten()
            x = np.array(data.x)
            y = np.array(data.y)
            d = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
            return current, d
        return None, None

    def _pull_atom_back(self):
        """Pull atom to center of limits."""
        pos0 = self.atom_absolute_nm
        center_x = np.mean(self.manip_limit_nm[:2]) + 2 * np.random.random() - 1
        center_y = np.mean(self.manip_limit_nm[2:]) + 2 * np.random.random() - 1

        self.transport.lateral_manipulation(
            x_start_nm=pos0[0],
            y_start_nm=pos0[1],
            x_end_nm=center_x,
            y_end_nm=center_y,
            bias_mv=self.pull_back_mV,
            current_pa=self.pull_back_pA,
            offset_nm=self.offset_nm,
            size_nm=self.len_nm,
        )

    # ── Helpers (ported from RealExpEnv) ──────────────────────

    def _action_to_manip_params(self, action):
        x_s = action[0] * self.step_nm
        y_s = action[1] * self.step_nm
        x_e = action[2] * self.goal_nm
        y_e = action[3] * self.goal_nm
        mvolt = np.clip(action[4], a_min=None, a_max=1) * self.max_mvolt
        pcurrent = (
            np.clip(action[5], a_min=None, a_max=1)
            * self.max_pcurrent_to_mvolt_ratio
            * mvolt
        )
        return x_s, y_s, x_e, y_e, mvolt, pcurrent

    def _compute_reward(self, state, next_state):
        old_p, _ = self._potential(state)
        new_p, dist = self._potential(next_state)
        reward = (
            self.default_reward_done * (dist < self.precision_lim)
            + self.default_reward * (dist > self.precision_lim)
            + new_p
            - old_p
        )
        return reward

    def _potential(self, state):
        dist = np.linalg.norm(
            state[:2] * self.goal_nm - state[2:] * self.goal_nm
        )
        return -dist / self.lattice_constant, dist

    def _get_destination(self, atom_rel, atom_abs, goal_nm):
        while True:
            angle = 2 * np.pi * np.random.random()
            dr = goal_nm * np.array([np.cos(angle), np.sin(angle)])
            dest_abs = atom_abs + dr
            if not self.out_of_range(dest_abs, self.inner_limit_nm):
                break
        dest_rel = atom_rel + dr
        return dest_rel, dest_abs, dr / self.goal_nm

    def _detect_current_jump(self, current):
        if current is None:
            return False
        if self._atom_move_detector is not None:
            success, prediction = self._atom_move_detector.predict(current)
            if success:
                return True
        # Fallback: gradient-based
        try:
            import findiff
            diff = findiff.FinDiff(0, 1, acc=6)(current)[3:-3]
            return np.sum(np.abs(diff) > self.current_jump * np.std(current)) > 2
        except ImportError:
            diff = np.gradient(current)
            return np.sum(np.abs(diff) > self.current_jump * np.std(current)) > 2

    @staticmethod
    def out_of_range(nm, limit_nm):
        if limit_nm is None:
            return False
        return np.any(
            (nm - limit_nm[[0, 2]]) * (nm - limit_nm[[1, 3]]) > 0, axis=-1
        )
