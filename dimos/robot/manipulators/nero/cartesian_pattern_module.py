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

"""Runtime-controllable cartesian pattern generator for the NERO arms.

A DimOS ``Module`` that continuously streams an end-effector pose to a NERO
``cartesian_ik`` coordinator, producing a repeating "poke" (oscillation along a
world axis) around a movable reference centre. The centre and the motion
parameters (period, amplitude, sweep) are exposed as ``@rpc`` methods, so an
external controller can move the reference point and reshape the motion at
runtime. These same methods are the natural hook for the DimOS skills/agent
system later: decorate them as ``@skill`` and add ``McpServer`` to drive them
with natural language.

Data flow (autoconnected in the blueprint):

    coordinator ── Out[JointState] coordinator_joint_state ──► this module (seed centre via FK)
    this module ── Out[PoseStamped] coordinator_cartesian_command ──► coordinator (cartesian_ik task)

Targets are generated in the WORLD frame and transformed into the arm base frame
before publishing (the control-side Pinocchio model has base_link at the origin).

Safety: this path has no collision or joint-limit avoidance. Keep the centre and
amplitude/sweep inside a known-safe workspace.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
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
from dimos.utils.logging_config import setup_logger
from dimos.utils.transform_utils import matrix_to_pose, pose_to_matrix

logger = setup_logger()

_AXES = {"x": 0, "y": 1, "z": 2}


class NeroCartesianPatternConfig(ModuleConfig):
    """Configuration for the NERO cartesian pattern generator.

    Attributes:
        arm: Which arm to drive ("left_arm" or "right_arm"). Determines the
            cartesian_ik task name, base pose, and joint names.
        amplitude: Poke half-stroke in metres (peak offset from the centre).
        period: Seconds per poke cycle.
        rate_hz: Publish rate.
        poke_axis: World axis the poke oscillates along ("x"/"y"/"z").
        sweep_axis: World axis the centre auto-drifts along.
        sweep_speed: Centre drift speed (m/s) along sweep_axis (0 = stationary).
        active: Start streaming immediately when True.
    """

    arm: str = "left_arm"
    amplitude: float = 0.03
    period: float = 2.0
    rate_hz: float = 50.0
    poke_axis: str = "z"
    sweep_axis: str = "x"
    sweep_speed: float = 0.02
    active: bool = True


class NeroCartesianPatternModule(Module):
    """Streams a movable, parameterized poke pattern to a NERO cartesian_ik task."""

    config: NeroCartesianPatternConfig

    coordinator_joint_state: In[JointState]
    coordinator_cartesian_command: Out[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        arm = self.config.arm
        if arm == "left_arm":
            self._task_name = "cartesian_ik_left_arm"
            bp = base_pose(
                x=NERO_BASE_X, y=NERO_LEFT_BASE_Y, z=NERO_BASE_Z,
                roll=NERO_LEFT_BASE_RPY[0], pitch=NERO_LEFT_BASE_RPY[1], yaw=NERO_LEFT_BASE_RPY[2],
            )
        elif arm == "right_arm":
            self._task_name = "cartesian_ik_right_arm"
            bp = base_pose(
                x=NERO_BASE_X, y=NERO_RIGHT_BASE_Y, z=NERO_BASE_Z,
                roll=NERO_RIGHT_BASE_RPY[0], pitch=NERO_RIGHT_BASE_RPY[1],
                yaw=NERO_RIGHT_BASE_RPY[2],
            )
        else:
            raise ValueError(f"arm must be 'left_arm' or 'right_arm', got {arm!r}")

        self._t_world_base = pose_to_matrix(bp)
        self._t_base_world = np.linalg.inv(self._t_world_base)
        self._joint_names = [f"{arm}/joint{i}" for i in range(1, NERO_DOF + 1)]
        self._ik = PinocchioIK.from_model_path(NERO_FK_MODEL, NERO_EE_JOINT_ID)

        # Mutable runtime state (guarded by _lock).
        self._center: np.ndarray | None = None  # world-frame [x,y,z]
        self._orientation: Quaternion | None = None  # held world orientation
        self._latest_q: np.ndarray | None = None
        self._amplitude = float(self.config.amplitude)
        self._period = max(1e-3, float(self.config.period))
        self._sweep_speed = float(self.config.sweep_speed)
        self._active = bool(self.config.active)
        self._poke_idx = _AXES[self.config.poke_axis]
        self._sweep_idx = _AXES[self.config.sweep_axis]

    # ------------------------------------------------------------------ lifecycle
    @rpc
    def start(self) -> None:
        super().start()
        self._stop.clear()
        self.coordinator_joint_state.subscribe(self._on_joint_state)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("NeroCartesianPatternModule started", arm=self.config.arm)

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    # ------------------------------------------------------------------ runtime control (RPC)
    @rpc
    def set_center(self, x: float, y: float, z: float) -> str:
        """Move the reference centre to an absolute world position (metres)."""
        with self._lock:
            self._center = np.array([float(x), float(y), float(z)], dtype=float)
        return f"center set to ({x:.3f}, {y:.3f}, {z:.3f})"

    @rpc
    def nudge(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
        """Shift the reference centre by a world-frame delta (metres)."""
        with self._lock:
            if self._center is None:
                return "center not seeded yet"
            self._center = self._center + np.array([dx, dy, dz], dtype=float)
            c = self._center.tolist()
        return f"center nudged to ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})"

    @rpc
    def set_period(self, seconds: float) -> str:
        """Set the poke cycle period in seconds (higher = slower)."""
        with self._lock:
            self._period = max(1e-3, float(seconds))
        return f"period set to {self._period:.3f}s"

    @rpc
    def set_amplitude(self, meters: float) -> str:
        """Set the poke half-stroke amplitude in metres."""
        with self._lock:
            self._amplitude = float(meters)
        return f"amplitude set to {self._amplitude:.3f}m"

    @rpc
    def set_sweep_speed(self, mps: float) -> str:
        """Set the centre auto-drift speed along the sweep axis (m/s, 0 = stop)."""
        with self._lock:
            self._sweep_speed = float(mps)
        return f"sweep speed set to {self._sweep_speed:.3f}m/s"

    @rpc
    def set_active(self, on: bool) -> str:
        """Enable/disable streaming. When off, the task times out and holds."""
        with self._lock:
            self._active = bool(on)
        return f"active={self._active}"

    @rpc
    def get_center(self) -> list[float]:
        """Return the current world-frame reference centre (or [] if unseeded)."""
        with self._lock:
            return [] if self._center is None else self._center.tolist()

    # ------------------------------------------------------------------ internals
    def _on_joint_state(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position, strict=False))
        try:
            q = np.array([by_name[n] for n in self._joint_names], dtype=float)
        except KeyError:
            return  # this message isn't for our arm / incomplete
        with self._lock:
            self._latest_q = q

    def _try_seed(self) -> None:
        """Seed centre + held orientation from the current EE (FK) once joints arrive."""
        q = self._latest_q
        if q is None:
            return
        se3 = self._ik.forward_kinematics(q)  # base-frame EE
        t_base_ee = np.asarray(se3.homogeneous, dtype=float)
        world = matrix_to_pose(self._t_world_base @ t_base_ee)
        self._center = np.array([world.position.x, world.position.y, world.position.z], dtype=float)
        self._orientation = Quaternion(
            world.orientation.x, world.orientation.y, world.orientation.z, world.orientation.w
        )
        logger.info("Pattern centre seeded from current EE", center=self._center.tolist())

    def _loop(self) -> None:
        dt = 1.0 / max(1e-3, self.config.rate_hz)
        t0 = time.perf_counter()
        last = t0
        while not self._stop.is_set():
            now = time.perf_counter()
            elapsed = now - last
            last = now
            cmd: PoseStamped | None = None
            with self._lock:
                if self._center is None:
                    self._try_seed()
                if self._active and self._center is not None and self._orientation is not None:
                    # Auto-drift the centre along the sweep axis.
                    self._center[self._sweep_idx] += self._sweep_speed * elapsed
                    phase = 2.0 * np.pi * ((now - t0) / self._period)
                    offset = np.zeros(3)
                    offset[self._poke_idx] = self._amplitude * np.sin(phase)
                    world_pos = self._center + offset
                    orientation = self._orientation
                    cmd = self._to_base_command(world_pos, orientation)
            if cmd is not None:
                self.coordinator_cartesian_command.publish(cmd)
            time.sleep(dt)

    def _to_base_command(self, world_pos: np.ndarray, orientation: Quaternion) -> PoseStamped:
        world_pose = Pose(
            position=Vector3(x=world_pos[0], y=world_pos[1], z=world_pos[2]),
            orientation=orientation,
        )
        base = matrix_to_pose(self._t_base_world @ pose_to_matrix(world_pose))
        return PoseStamped(
            frame_id=self._task_name,
            position=base.position,
            orientation=base.orientation,
        )
