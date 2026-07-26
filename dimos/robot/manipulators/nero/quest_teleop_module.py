# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bimanual Quest -> NERO cartesian teleop bridge (trigger-to-engage).

Single source of truth for teleop engagement and motion. Consumes the absolute
robot-frame controller poses from ``NeroControllerStreamModule`` plus the
per-hand trigger from ``teleop_buttons``, and streams world->base cartesian
commands to the shared ``coordinator-nero-cartesian-*`` coordinators on
``/coordinator_cartesian_command`` (frame_id routes each arm).

Engagement model (per arm, independent):
  * left controller trigger  -> left arm   (unless swap_controllers)
  * right controller trigger -> right arm
  * On trigger rising edge: capture C0 (controller pose now) and A0 (the arm's
    current end-effector pose via FK, world frame). Re-anchoring on every pull
    means the absolute start location never matters and there is no jump on
    re-engage.
  * While held:
        target_pos = A0.pos + position_scale * M @ (C_now.pos - C0.pos)
        target_orn = (M R_delta M^T) * A0.orn      (when track_orientation)
    where M is the per-arm axis map and R_delta is the controller's rotation
    since engage.
  * On release / never pulled: the arm is stationary (last target is held).

Smoothness pipeline (why each stage exists)
-------------------------------------------
The raw controller signal is noisy (Quest tracking jitter + hand tremor) and
arrives at the stream module's ``control_loop_hz``, while commands are emitted
at the coordinator's joint-state rate. Feeding the raw signal straight to
``CartesianIKTask`` causes two failure modes seen on hardware:

  1. Jitter -- step changes in the target make the IK solution jump. If a
     solution's joint delta exceeds ``max_joint_delta_deg`` the task REJECTS it
     and emits nothing that tick (cartesian_ik_task.compute), so the arm stalls
     then lurches.
  2. Stuck poses -- the target is computed from the hand only, so if the arm
     cannot keep up (or the target leaves the reachable set) the target runs
     away from the actual end-effector. IK then never converges and the arm
     wedges, staying wedged even after the hand comes back.

So the target passes through, in order:
  deadband -> workspace clamp -> low-pass (time-constant EMA / slerp)
  -> velocity + angular-velocity limit -> tracking-error leash

The leash is the important one for "stuck": the commanded target is never
allowed to sit further than ``max_tracking_error_m`` from the arm's measured
end-effector, so IK always has a nearby, feasible goal and recovers on its own.

Safety: this path performs NO collision or joint-limit avoidance beyond the
cartesian_ik task's per-tick joint-delta clamp. ``max_offset_m`` bounds the
workspace, the leash bounds tracking error, and trigger release is the deadman.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.manipulation.planning.kinematics.pinocchio_ik import PinocchioIK
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators._modeling import base_pose
from dimos.robot.manipulators.nero.config import (
    NERO_BASE_X,
    NERO_BASE_Z,
    NERO_DOF,
    NERO_EE_JOINT_ID,
    NERO_FK_MODEL,
    NERO_LEFT_BASE_RPY,
    NERO_LEFT_BASE_Y,
    NERO_RIGHT_BASE_RPY,
    NERO_RIGHT_BASE_Y,
)
from dimos.teleop.quest.quest_types import Buttons, Hand
from dimos.utils.logging_config import setup_logger
from dimos.utils.transform_utils import matrix_to_pose, pose_to_matrix

logger = setup_logger()

_TELEMETRY_LOG_PERIOD_S = 0.5
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return _IDENTITY_QUAT.copy()
    return q / n


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc slerp between two xyzw quaternions."""
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the shorter path
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly aligned -> linear is numerically safer
        return _normalize_quat(q0 + t * (q1 - q0))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin((1.0 - t) * theta)) / sin_theta
    s1 = float(np.sin(t * theta)) / sin_theta
    return _normalize_quat(s0 * q0 + s1 * q1)


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Absolute rotation angle (radians) between two xyzw quaternions."""
    dot = abs(float(np.dot(_normalize_quat(q0), _normalize_quat(q1))))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


@dataclass
class _ArmState:
    """Per-arm teleop state (guarded by the module lock)."""

    arm: str
    task_name: str
    t_world_base: np.ndarray
    t_base_world: np.ndarray
    joint_names: list[str]
    axis_idx: np.ndarray                      # (3,) input controller-axis per world axis
    axis_sign: np.ndarray                     # (3,) sign per world axis
    axis_matrix: np.ndarray                   # (3,3) M, for rotation conjugation

    latest_q: np.ndarray | None = None        # current joints for this arm
    controller_pos: np.ndarray | None = None  # C_now position
    controller_orn: np.ndarray = field(default_factory=lambda: _IDENTITY_QUAT.copy())
    engaged: bool = False
    prev_trigger: bool = False
    c0: np.ndarray | None = None              # controller pos at engage
    c0_orn: np.ndarray | None = None          # controller orientation at engage
    a0_pos: np.ndarray | None = None          # world EE pos at engage
    a0_orn: np.ndarray | None = None          # world EE orientation at engage (xyzw)
    target_pos: np.ndarray | None = None      # smoothed, published world position
    target_orn: np.ndarray | None = None      # smoothed, published world orientation
    last_target: PoseStamped | None = None    # last published command (for hold)
    reject_streak: int = 0                    # consecutive leash saturations


class NeroBimanualQuestTeleopConfig(ModuleConfig):
    """Configuration for the bimanual Quest teleop bridge.

    Attributes:
        position_scale: Controller metres -> world metres (0.5 = arm moves half).
        max_offset_m: Clamp on |offset| from the anchor (workspace guard).
        trigger_threshold: Analog trigger value [0,1] above which a hand engages.
        motion_enabled: If False, engage/anchor is logged but NO command is
            published (arm stays put).
        swap_controllers: If True, the LEFT controller drives the RIGHT arm and
            vice versa.
        left_axis_map / right_axis_map: How controller-delta axes map to the
            world-offset axes for each arm, as 3 signed labels [world_x, world_y,
            world_z]. Each label is one of "x","y","z","-x","-y","-z" naming the
            controller-delta axis (and sign) that drives that world axis. The
            same map is applied to orientation by conjugation, so rotation stays
            consistent with translation.
        track_orientation: Track the controller's rotation as well as position.
            Recommended: holding a fixed orientation while translating
            over-constrains the arm and is a common cause of unreachable targets
            and wedged poses.
        rotation_scale: Fraction of the controller's rotation to apply (1.0 =
            1:1, 0.5 = half). Lower is calmer.
        smoothing_tau_s: Low-pass time constant (seconds) applied to the target
            position and orientation. Higher = smoother but laggier. 0 disables.
        max_speed_mps: Maximum commanded target speed (m/s).
        max_angular_speed_dps: Maximum commanded target angular speed (deg/s).
        max_tracking_error_m: The leash. The published target may never be
            further than this from the arm's measured end-effector position.
            Prevents target runaway, keeps IK feasible, and lets a saturated arm
            recover instead of wedging.
        deadband_m: Ignore controller position changes smaller than this
            (kills sub-millimetre tracking dither). 0 disables.
        command_rate_hz: Throttle for published cartesian commands. 0 = publish
            every joint-state tick (~100 Hz), which suits the CPV backend. The
            move_j backend is a driver-planned move, so flooding it at 100 Hz
            makes it re-plan constantly; throttle to ~20-30 Hz there. Keep well
            above 2 Hz or the cartesian_ik task's 0.5 s timeout will trip.
    """

    position_scale: float = 0.5
    max_offset_m: float = 0.30
    trigger_threshold: float = 0.5
    motion_enabled: bool = False
    swap_controllers: bool = False
    left_axis_map: list[str] = Field(default_factory=lambda: ["x", "y", "z"])
    right_axis_map: list[str] = Field(default_factory=lambda: ["x", "y", "z"])
    # --- smoothness / robustness ---
    track_orientation: bool = True
    rotation_scale: float = 1.0
    smoothing_tau_s: float = 0.08
    max_speed_mps: float = 0.35
    max_angular_speed_dps: float = 120.0
    max_tracking_error_m: float = 0.12
    deadband_m: float = 0.001
    command_rate_hz: float = 0.0


class NeroBimanualQuestTeleopModule(Module):
    """Turns engaged Quest controller motion into smooth NERO cartesian commands."""

    config: NeroBimanualQuestTeleopConfig

    left_controller_output: In[PoseStamped]
    right_controller_output: In[PoseStamped]
    teleop_buttons: In[Buttons]
    coordinator_joint_state: In[JointState]
    coordinator_cartesian_command: Out[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_telemetry_log = 0.0
        self._last_tick_time = 0.0
        self._last_publish_time = 0.0
        # Diagnostics visible from captured (subscription) threads.
        self._tick = 0
        self._publish_count = 0

        # One FK/IK model shared by both arms (base_link at origin in the model).
        self._ik = PinocchioIK.from_model_path(NERO_FK_MODEL, NERO_EE_JOINT_ID)

        left_arm_state = self._make_arm(
            "left_arm", "cartesian_ik_left_arm", NERO_LEFT_BASE_Y, NERO_LEFT_BASE_RPY,
            self.config.left_axis_map,
        )
        right_arm_state = self._make_arm(
            "right_arm", "cartesian_ik_right_arm", NERO_RIGHT_BASE_Y, NERO_RIGHT_BASE_RPY,
            self.config.right_axis_map,
        )
        # Map each controller hand to the arm it drives (optionally swapped).
        if self.config.swap_controllers:
            self._arms: dict[Hand, _ArmState] = {
                Hand.LEFT: right_arm_state,
                Hand.RIGHT: left_arm_state,
            }
        else:
            self._arms = {Hand.LEFT: left_arm_state, Hand.RIGHT: right_arm_state}

    @staticmethod
    def _parse_axis_map(axis_map: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Turn ["x","-y","z"] into (indices, signs, matrix M) for remapping.

        M is the 3x3 form of the same map, used to conjugate rotations
        (R_world = M R_ctrl M^T) so orientation tracking stays consistent with
        the translation mapping.
        """
        axes = {"x": 0, "y": 1, "z": 2}
        if len(axis_map) != 3:
            raise ValueError(f"axis_map must have 3 entries, got {axis_map!r}")
        idx = np.zeros(3, dtype=int)
        sign = np.ones(3, dtype=float)
        for i, label in enumerate(axis_map):
            s = label.strip().lower()
            if s.startswith("-"):
                sign[i] = -1.0
                s = s[1:]
            elif s.startswith("+"):
                s = s[1:]
            if s not in axes:
                raise ValueError(f"axis_map entry {label!r} invalid; use x/y/z with optional sign")
            idx[i] = axes[s]
        if len(set(idx.tolist())) != 3:
            raise ValueError(f"axis_map must use each of x/y/z exactly once, got {axis_map!r}")
        matrix = np.zeros((3, 3), dtype=float)
        for i in range(3):
            matrix[i, idx[i]] = sign[i]
        return idx, sign, matrix

    @classmethod
    def _make_arm(cls, arm: str, task: str, base_y: float,
                  base_rpy: tuple[float, float, float], axis_map: list[str]) -> _ArmState:
        bp = base_pose(
            x=NERO_BASE_X, y=base_y, z=NERO_BASE_Z,
            roll=base_rpy[0], pitch=base_rpy[1], yaw=base_rpy[2],
        )
        t_world_base = pose_to_matrix(bp)
        idx, sign, matrix = cls._parse_axis_map(axis_map)
        return _ArmState(
            arm=arm,
            task_name=task,
            t_world_base=t_world_base,
            t_base_world=np.linalg.inv(t_world_base),
            joint_names=[f"{arm}/joint{i}" for i in range(1, NERO_DOF + 1)],
            axis_idx=idx,
            axis_sign=sign,
            axis_matrix=matrix,
        )

    # ------------------------------------------------------------------ lifecycle
    @rpc
    def start(self) -> None:
        super().start()
        self._stop.clear()
        self.left_controller_output.subscribe(self._on_left_pose)
        self.right_controller_output.subscribe(self._on_right_pose)
        self.teleop_buttons.subscribe(self._on_buttons)
        # Commands are generated in this callback (fires at the coordinator's
        # joint-state rate, ~100 Hz) rather than a hand-rolled thread, because
        # modules run as Dask actors and a bare threading.Thread is not
        # reliably scheduled in the actor process.
        self.coordinator_joint_state.subscribe(self._on_joint_state)
        logger.info(
            "NeroBimanualQuestTeleopModule started",
            motion_enabled=self.config.motion_enabled,
            position_scale=self.config.position_scale,
            trigger_threshold=self.config.trigger_threshold,
            track_orientation=self.config.track_orientation,
            smoothing_tau_s=self.config.smoothing_tau_s,
            max_tracking_error_m=self.config.max_tracking_error_m,
        )

    @rpc
    def stop(self) -> None:
        self._stop.set()
        super().stop()

    # ------------------------------------------------------------------ runtime tuning (RPC)
    @rpc
    def set_position_scale(self, scale: float) -> str:
        """Set the controller->world position scale (lower = calmer)."""
        self.config.position_scale = float(scale)
        return f"position_scale={self.config.position_scale:.3f}"

    @rpc
    def set_smoothing(self, tau_s: float) -> str:
        """Set the low-pass time constant in seconds (higher = smoother)."""
        self.config.smoothing_tau_s = max(0.0, float(tau_s))
        return f"smoothing_tau_s={self.config.smoothing_tau_s:.3f}"

    @rpc
    def set_track_orientation(self, on: bool) -> str:
        """Enable/disable controller orientation tracking."""
        self.config.track_orientation = bool(on)
        return f"track_orientation={self.config.track_orientation}"

    @rpc
    def set_max_tracking_error(self, meters: float) -> str:
        """Set the leash: max distance the target may sit from the actual EE."""
        self.config.max_tracking_error_m = max(0.01, float(meters))
        return f"max_tracking_error_m={self.config.max_tracking_error_m:.3f}"

    @rpc
    def get_state(self) -> str:
        """Return a one-line summary of both arms' teleop state."""
        with self._lock:
            return (
                f"tick={self._tick} publishes={self._publish_count} "
                f"left=[{self._describe(self._arms[Hand.LEFT])}] "
                f"right=[{self._describe(self._arms[Hand.RIGHT])}]"
            )

    # ------------------------------------------------------------------ inputs
    def _on_left_pose(self, msg: PoseStamped) -> None:
        self._store_controller_pose(Hand.LEFT, msg)

    def _on_right_pose(self, msg: PoseStamped) -> None:
        self._store_controller_pose(Hand.RIGHT, msg)

    def _store_controller_pose(self, hand: Hand, msg: PoseStamped) -> None:
        pos = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=float)
        orn = np.array(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            dtype=float,
        )
        with self._lock:
            arm = self._arms[hand]
            arm.controller_pos = pos
            arm.controller_orn = _normalize_quat(orn)

    def _on_buttons(self, msg: Buttons) -> None:
        # Analog is authoritative; fall back to the digital bit (set when the
        # publisher saw trigger > 0.5) so engagement still works if analog
        # packing is ever absent.
        analog = {Hand.LEFT: msg.left_trigger_analog, Hand.RIGHT: msg.right_trigger_analog}
        digital = {Hand.LEFT: bool(msg.left_trigger), Hand.RIGHT: bool(msg.right_trigger)}
        with self._lock:
            for hand, arm in self._arms.items():
                pressed = analog[hand] >= self.config.trigger_threshold or digital[hand]
                if pressed and not arm.prev_trigger:
                    self._engage_locked(arm)
                elif not pressed and arm.prev_trigger:
                    self._disengage_locked(arm)
                arm.prev_trigger = pressed

    def _on_joint_state(self, msg: JointState) -> None:
        """Update joints and emit cartesian commands for engaged arms.

        Fires at the coordinator's joint-state publish rate (~100 Hz), so this
        doubles as the command streaming clock.
        """
        now = time.perf_counter()
        dt = now - self._last_tick_time if self._last_tick_time > 0.0 else 0.01
        self._last_tick_time = now
        dt = float(np.clip(dt, 1e-3, 0.1))  # ignore pathological gaps

        by_name = dict(zip(msg.name, msg.position, strict=False))
        cmds: list[PoseStamped] = []
        with self._lock:
            self._tick += 1
            # The target is always integrated at the full tick rate (so filters
            # and limits stay smooth); only publishing is throttled.
            rate = self.config.command_rate_hz
            may_publish = True
            if rate > 0.0:
                if now - self._last_publish_time < (1.0 / rate):
                    may_publish = False
                else:
                    self._last_publish_time = now
            for arm in self._arms.values():
                try:
                    arm.latest_q = np.array([by_name[n] for n in arm.joint_names], dtype=float)
                except KeyError:
                    pass  # this message doesn't carry this arm's joints
                cmd = self._compute_command_locked(arm, dt)
                if cmd is not None and may_publish:
                    cmds.append(cmd)
            self._log_telemetry_locked()
        for cmd in cmds:
            self.coordinator_cartesian_command.publish(cmd)
            self._publish_count += 1

    # ------------------------------------------------------------------ engage
    def _ee_world_locked(self, arm: _ArmState) -> tuple[np.ndarray, np.ndarray] | None:
        """Current end-effector pose in the world frame, via FK. (pos, xyzw)."""
        if arm.latest_q is None:
            return None
        se3 = self._ik.forward_kinematics(arm.latest_q)
        world = matrix_to_pose(arm.t_world_base @ np.asarray(se3.homogeneous, dtype=float))
        pos = np.array([world.position.x, world.position.y, world.position.z], dtype=float)
        orn = _normalize_quat(
            np.array(
                [
                    world.orientation.x,
                    world.orientation.y,
                    world.orientation.z,
                    world.orientation.w,
                ],
                dtype=float,
            )
        )
        return pos, orn

    def _engage_locked(self, arm: _ArmState) -> None:
        ee = self._ee_world_locked(arm)
        if ee is None:
            logger.warning(f"{arm.arm}: engage ignored, no joint state yet")
            return
        if arm.controller_pos is None:
            logger.warning(f"{arm.arm}: engage ignored, no controller pose yet")
            return
        arm.a0_pos, arm.a0_orn = ee
        arm.c0 = arm.controller_pos.copy()
        arm.c0_orn = arm.controller_orn.copy()
        # Start the smoothed target exactly at the current EE so there is no
        # jump on engage, and reset saturation bookkeeping.
        arm.target_pos = arm.a0_pos.copy()
        arm.target_orn = arm.a0_orn.copy()
        arm.reject_streak = 0
        arm.engaged = True
        logger.info(
            f"{arm.arm}: ENGAGED",
            anchor_world=np.round(arm.a0_pos, 4).tolist(),
            controller_c0=np.round(arm.c0, 4).tolist(),
        )

    def _disengage_locked(self, arm: _ArmState) -> None:
        arm.engaged = False
        logger.info(f"{arm.arm}: released (holding)")

    # ------------------------------------------------------------------ command generation
    def _desired_pose_locked(self, arm: _ArmState) -> tuple[np.ndarray, np.ndarray] | None:
        """Raw (unsmoothed) desired world pose from the controller, or None."""
        if arm.a0_pos is None or arm.a0_orn is None or arm.c0 is None:
            return None
        if arm.controller_pos is None:
            return None

        raw = arm.controller_pos - arm.c0
        # Deadband: suppress sub-millimetre tracking dither while holding still.
        if self.config.deadband_m > 0.0:
            small = np.abs(raw) < self.config.deadband_m
            raw = np.where(small, 0.0, raw)

        offset = arm.axis_sign * raw[arm.axis_idx] * self.config.position_scale
        norm = float(np.linalg.norm(offset))
        if norm > self.config.max_offset_m:  # workspace guard
            offset *= self.config.max_offset_m / norm
        desired_pos = arm.a0_pos + offset

        desired_orn = arm.a0_orn
        if self.config.track_orientation and arm.c0_orn is not None:
            # Controller rotation since engage, in the controller frame.
            r_now = Rotation.from_quat(arm.controller_orn)
            r_c0 = Rotation.from_quat(arm.c0_orn)
            r_delta = r_now * r_c0.inv()
            # Optionally apply only a fraction of the rotation.
            if self.config.rotation_scale != 1.0:
                axis_angle = r_delta.as_rotvec() * float(self.config.rotation_scale)
                r_delta = Rotation.from_rotvec(axis_angle)
            # Express the delta in the world frame using the same axis map as
            # translation: R_world = M R_ctrl M^T.
            m = arm.axis_matrix
            r_world_delta = Rotation.from_matrix(m @ r_delta.as_matrix() @ m.T)
            desired_orn = _normalize_quat((r_world_delta * Rotation.from_quat(arm.a0_orn)).as_quat())

        return desired_pos, desired_orn

    def _compute_command_locked(self, arm: _ArmState, dt: float) -> PoseStamped | None:
        """Return the command to publish for this arm this tick, or None.

        Pipeline: desired -> low-pass -> velocity limit -> tracking leash.
        When disengaged the last target is held (and still published) so the
        cartesian_ik task never times out mid-motion and the arm holds position.
        """
        if not self.config.motion_enabled:
            return None

        if arm.engaged:
            desired = self._desired_pose_locked(arm)
            if desired is None:
                return arm.last_target
            desired_pos, desired_orn = desired
        elif arm.target_pos is not None and arm.target_orn is not None:
            # Held: desired == current target, so the filters are a no-op.
            desired_pos, desired_orn = arm.target_pos, arm.target_orn
        else:
            return arm.last_target

        if arm.target_pos is None or arm.target_orn is None:
            arm.target_pos = desired_pos.copy()
            arm.target_orn = desired_orn.copy()

        # 1. Low-pass filter with a rate-independent time constant.
        tau = self.config.smoothing_tau_s
        alpha = 1.0 if tau <= 0.0 else 1.0 - float(np.exp(-dt / tau))
        new_pos = arm.target_pos + alpha * (desired_pos - arm.target_pos)
        new_orn = _slerp(arm.target_orn, desired_orn, alpha)

        # 2. Velocity limit -- bounds how fast the IK target can move, which is
        #    what keeps solutions inside the task's per-tick joint-delta clamp.
        step = new_pos - arm.target_pos
        max_step = self.config.max_speed_mps * dt
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step > 0.0:
            step *= max_step / step_norm
        new_pos = arm.target_pos + step

        # 3. Angular velocity limit.
        max_rad = np.radians(self.config.max_angular_speed_dps) * dt
        angle = _quat_angle(arm.target_orn, new_orn)
        if angle > max_rad > 0.0:
            new_orn = _slerp(arm.target_orn, new_orn, max_rad / angle)

        # 4. Tracking-error leash. The target may never sit further than
        #    max_tracking_error_m from where the arm actually is. This stops the
        #    target running away when the arm saturates or the goal is
        #    unreachable, which is what otherwise leaves the arm wedged.
        ee = self._ee_world_locked(arm)
        if ee is not None:
            ee_pos, _ = ee
            err = new_pos - ee_pos
            err_norm = float(np.linalg.norm(err))
            leash = self.config.max_tracking_error_m
            if err_norm > leash > 0.0:
                new_pos = ee_pos + err * (leash / err_norm)
                arm.reject_streak += 1
                if arm.reject_streak % 100 == 1:
                    logger.warning(
                        f"{arm.arm}: target leashed (arm not keeping up / unreachable)",
                        tracking_error_m=round(err_norm, 4),
                        leash_m=leash,
                    )
            else:
                arm.reject_streak = 0

        arm.target_pos = new_pos
        arm.target_orn = _normalize_quat(new_orn)
        arm.last_target = self._to_base_command(arm, arm.target_pos, arm.target_orn)
        return arm.last_target

    def _log_telemetry_locked(self) -> None:
        now = time.perf_counter()
        if now - self._last_telemetry_log < _TELEMETRY_LOG_PERIOD_S:
            return
        self._last_telemetry_log = now
        logger.info(
            "teleop state",
            tick=self._tick,
            publishes=self._publish_count,
            left=self._describe(self._arms[Hand.LEFT]),
            right=self._describe(self._arms[Hand.RIGHT]),
        )

    @staticmethod
    def _describe(arm: _ArmState) -> str:
        eng = "ENGAGED" if arm.engaged else "hold"
        tgt = "none" if arm.target_pos is None else np.round(arm.target_pos, 3).tolist()
        leash = f" leashed({arm.reject_streak})" if arm.reject_streak else ""
        return f"{eng} target={tgt}{leash}"

    def _to_base_command(self, arm: _ArmState, world_pos: np.ndarray,
                         world_orn: np.ndarray) -> PoseStamped:
        world_pose = Pose(
            position=Vector3(x=world_pos[0], y=world_pos[1], z=world_pos[2]),
            orientation=Quaternion(world_orn[0], world_orn[1], world_orn[2], world_orn[3]),
        )
        base = matrix_to_pose(arm.t_base_world @ pose_to_matrix(world_pose))
        return PoseStamped(
            frame_id=arm.task_name,
            position=base.position,
            orientation=base.orientation,
        )
