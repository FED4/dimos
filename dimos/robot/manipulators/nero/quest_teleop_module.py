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
  * left controller trigger  -> left arm
  * right controller trigger -> right arm
  * On trigger rising edge: capture C0 (controller position now) and A0 (the
    arm's current end-effector pose via FK, in world frame). This re-anchors on
    every pull, so the absolute start location never matters and there is no
    jump on re-engage.
  * While held (position-only for now):
        target_world = A0.position + position_scale * (C_now - C0)
    with |offset| clamped to max_offset_m. Orientation is HELD at A0.
  * On release / never pulled: the arm is stationary.

Motion is gated by ``motion_enabled``. With it False (Step 2), the module logs
engage/anchor events and computes nothing on the wire -- useful for verifying FK
seeding and trigger edges with zero arm motion. Step 3 sets it True.

Safety: this path performs NO collision or joint-limit avoidance beyond the
cartesian_ik task's per-tick joint-delta clamp. ``max_offset_m`` is the
workspace guard; trigger release is the deadman.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np
from pydantic import Field

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

    latest_q: np.ndarray | None = None       # current joints for this arm
    controller_pos: np.ndarray | None = None  # C_now, robot/world frame
    engaged: bool = False
    prev_trigger: bool = False
    c0: np.ndarray | None = None              # controller pos at engage
    a0_pos: np.ndarray | None = None          # world EE pos at engage
    a0_orn: Quaternion | None = None          # world EE orientation, held
    last_target: PoseStamped | None = None    # last published command (for hold)


class NeroBimanualQuestTeleopConfig(ModuleConfig):
    """Configuration for the bimanual Quest teleop bridge.

    Attributes:
        position_scale: Controller metres -> world metres (0.5 = arm moves half).
        max_offset_m: Clamp on |offset| from the anchor (workspace guard).
        trigger_threshold: Analog trigger value [0,1] above which a hand engages.
        motion_enabled: If False, engage/anchor is logged but NO command is
            published (arm stays put). Step 2 uses False; Step 3 uses True.
        swap_controllers: If True, the LEFT controller drives the RIGHT arm and
            vice versa. Use when the controller/arm sides feel mirrored.
        left_axis_map / right_axis_map: How controller-delta axes map to the
            world-offset axes for each arm, as 3 signed labels [world_x, world_y,
            world_z]. Each label is one of "x","y","z","-x","-y","-z" naming the
            controller-delta axis (and sign) that drives that world axis. E.g.
            ["x","-y","z"] flips left/right; ["y","x","z"] swaps forward<->side.
    """

    position_scale: float = 0.5
    max_offset_m: float = 0.30
    trigger_threshold: float = 0.5
    motion_enabled: bool = False
    swap_controllers: bool = False
    left_axis_map: list[str] = Field(default_factory=lambda: ["x", "y", "z"])
    right_axis_map: list[str] = Field(default_factory=lambda: ["x", "y", "z"])


class NeroBimanualQuestTeleopModule(Module):
    """Turns engaged Quest controller motion into NERO cartesian commands."""

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
    def _parse_axis_map(axis_map: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Turn ["x","-y","z"] into (indices, signs) arrays for fast remapping."""
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
        return idx, sign

    @classmethod
    def _make_arm(cls, arm: str, task: str, base_y: float,
                  base_rpy: tuple[float, float, float], axis_map: list[str]) -> _ArmState:
        bp = base_pose(
            x=NERO_BASE_X, y=base_y, z=NERO_BASE_Z,
            roll=base_rpy[0], pitch=base_rpy[1], yaw=base_rpy[2],
        )
        t_world_base = pose_to_matrix(bp)
        idx, sign = cls._parse_axis_map(axis_map)
        return _ArmState(
            arm=arm,
            task_name=task,
            t_world_base=t_world_base,
            t_base_world=np.linalg.inv(t_world_base),
            joint_names=[f"{arm}/joint{i}" for i in range(1, NERO_DOF + 1)],
            axis_idx=idx,
            axis_sign=sign,
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
        )

    @rpc
    def stop(self) -> None:
        self._stop.set()
        super().stop()

    # ------------------------------------------------------------------ inputs
    def _on_left_pose(self, msg: PoseStamped) -> None:
        self._store_controller_pos(Hand.LEFT, msg)

    def _on_right_pose(self, msg: PoseStamped) -> None:
        self._store_controller_pos(Hand.RIGHT, msg)

    def _store_controller_pos(self, hand: Hand, msg: PoseStamped) -> None:
        pos = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=float)
        with self._lock:
            self._arms[hand].controller_pos = pos

    def _on_joint_state(self, msg: JointState) -> None:
        """Update joints and emit cartesian commands for engaged arms.

        Fires at the coordinator's joint-state publish rate (~100 Hz), so this
        doubles as the command streaming clock.
        """
        by_name = dict(zip(msg.name, msg.position, strict=False))
        cmds: list[PoseStamped] = []
        with self._lock:
            self._tick += 1
            for arm in self._arms.values():
                try:
                    arm.latest_q = np.array([by_name[n] for n in arm.joint_names], dtype=float)
                except KeyError:
                    pass  # this message doesn't carry this arm's joints
                cmd = self._compute_command_locked(arm)
                if cmd is not None:
                    cmds.append(cmd)
            self._log_telemetry_locked()
        for cmd in cmds:
            self.coordinator_cartesian_command.publish(cmd)
            self._publish_count += 1

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

    # ------------------------------------------------------------------ engage
    def _engage_locked(self, arm: _ArmState) -> None:
        if arm.latest_q is None:
            logger.warning(f"{arm.arm}: engage ignored, no joint state yet")
            return
        if arm.controller_pos is None:
            logger.warning(f"{arm.arm}: engage ignored, no controller pose yet")
            return
        se3 = self._ik.forward_kinematics(arm.latest_q)
        world = matrix_to_pose(arm.t_world_base @ np.asarray(se3.homogeneous, dtype=float))
        arm.a0_pos = np.array([world.position.x, world.position.y, world.position.z], dtype=float)
        arm.a0_orn = Quaternion(
            world.orientation.x, world.orientation.y, world.orientation.z, world.orientation.w
        )
        arm.c0 = arm.controller_pos.copy()
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
    def _compute_command_locked(self, arm: _ArmState) -> PoseStamped | None:
        """Return the command to publish for this arm this tick, or None.

        Step 2 (motion_enabled False): always None -> no motion.
        Step 3 (motion_enabled True): tracks while engaged, holds last target
        after release so the task never times out mid-motion.
        """
        if not self.config.motion_enabled:
            return None
        if arm.engaged and arm.a0_pos is not None and arm.c0 is not None \
                and arm.controller_pos is not None and arm.a0_orn is not None:
            raw = arm.controller_pos - arm.c0
            # Remap controller-delta axes -> world-offset axes per this arm.
            remapped = arm.axis_sign * raw[arm.axis_idx]
            offset = remapped * self.config.position_scale
            n = float(np.linalg.norm(offset))
            if n > self.config.max_offset_m:
                offset *= self.config.max_offset_m / n
            world_pos = arm.a0_pos + offset
            arm.last_target = self._to_base_command(arm, world_pos, arm.a0_orn)
            return arm.last_target
        # Disengaged: keep streaming the last target so the arm holds in place.
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
        c = "?" if arm.controller_pos is None else np.round(arm.controller_pos, 3).tolist()
        a = "unseeded" if arm.a0_pos is None else np.round(arm.a0_pos, 3).tolist()
        return f"{eng} ctrl={c} anchor={a}"

    def _to_base_command(self, arm: _ArmState, world_pos: np.ndarray,
                         orientation: Quaternion) -> PoseStamped:
        world_pose = Pose(
            position=Vector3(x=world_pos[0], y=world_pos[1], z=world_pos[2]),
            orientation=orientation,
        )
        base = matrix_to_pose(arm.t_base_world @ pose_to_matrix(world_pose))
        return PoseStamped(
            frame_id=arm.task_name,
            position=base.position,
            orientation=base.orientation,
        )
