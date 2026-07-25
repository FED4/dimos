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

"""Runtime-controllable cartesian pattern planner for the NERO arms.

A DimOS ``Module`` that continuously streams an end-effector pose to a NERO
``cartesian_ik`` coordinator, producing a repeating pattern (``hold``/``line``/
``circle``) around a single movable reference centre. It is a standalone streamer:
run it as its own blueprint alongside a separately-launched cartesian coordinator
(``coordinator-nero-cartesian-mock`` / ``-left`` / ``-bimanual``). It publishes on
the shared ``/coordinator_cartesian_command`` LCM topic and reads the coordinator's
``/coordinator_joint_state`` to seed the centre via forward kinematics -- exactly
the two topics the coordinator already uses, so the coordinator is unchanged.

This unifies the two working NERO paths:
  * demo_cartesian_stream.py -- line/circle world-frame pattern generation.
  * NeroCartesianPatternModule -- @rpc live control + FK seeding + world->base.

The pattern set is a pluggable registry: the reference centre is moved with
``set_center``/``nudge`` (which moves the whole pattern), and the motion around it
is selected with ``set_pattern`` and shaped with ``set_amplitude``/``set_radius``/
``set_axis``/``set_plane``/``set_period``. Every ``@rpc`` here maps 1:1 to a future
``@skill`` for natural-language control.

Safety: this path has no collision or joint-limit avoidance. Keep the centre and
amplitude/radius inside a known-safe workspace and validate on the mock first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
_PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


@dataclass
class PatternParams:
    """Shape parameters read by the pattern functions each tick."""

    amplitude: float  # line half-stroke (metres)
    radius: float  # circle radius (metres)
    axis: int  # line axis index (world x/y/z -> 0/1/2)
    plane: tuple[int, int]  # circle plane axis indices


# Pluggable pattern registry: phase (radians) -> world-frame position offset (3,).
# The offset maths mirrors demo_cartesian_stream.py._offset so behaviour is unchanged.
PatternFn = Callable[[float, PatternParams], np.ndarray]
_PATTERNS: dict[str, PatternFn] = {}


def _register(name: str) -> Callable[[PatternFn], PatternFn]:
    def deco(fn: PatternFn) -> PatternFn:
        _PATTERNS[name] = fn
        return fn

    return deco


@_register("hold")
def _hold(phase: float, p: PatternParams) -> np.ndarray:
    """No motion -- hold at the reference centre (pure point control)."""
    return np.zeros(3)


@_register("line")
def _line(phase: float, p: PatternParams) -> np.ndarray:
    """Sinusoidal line along one world axis."""
    off = np.zeros(3)
    off[p.axis] = p.amplitude * np.sin(phase)
    return off


@_register("circle")
def _circle(phase: float, p: PatternParams) -> np.ndarray:
    """Circle in one world plane."""
    off = np.zeros(3)
    i, j = p.plane
    off[i] = p.radius * np.cos(phase)
    off[j] = p.radius * np.sin(phase)
    return off


class NeroPatternPlannerConfig(ModuleConfig):
    """Configuration for the NERO cartesian pattern planner.

    Attributes:
        arm: Which arm to drive ("left_arm" or "right_arm"). Determines the
            cartesian_ik task name, base pose, and joint names.
        pattern: Initial pattern ("hold", "line", or "circle").
        amplitude: Line half-stroke in metres (peak offset from the centre).
        radius: Circle radius in metres.
        period: Seconds per pattern cycle.
        rate_hz: Publish rate.
        axis: World axis the line oscillates along ("x"/"y"/"z").
        plane: World plane the circle traces ("xy"/"xz"/"yz").
        active: Start streaming immediately when True.
    """

    arm: str = "left_arm"
    pattern: str = "line"
    amplitude: float = 0.03
    radius: float = 0.04
    period: float = 2.0
    rate_hz: float = 100.0
    axis: str = "x"
    plane: str = "xz"
    active: bool = True


class NeroPatternPlannerModule(Module):
    """Streams a movable, switchable pattern to a NERO cartesian_ik task."""

    config: NeroPatternPlannerConfig

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
                x=NERO_BASE_X,
                y=NERO_LEFT_BASE_Y,
                z=NERO_BASE_Z,
                roll=NERO_LEFT_BASE_RPY[0],
                pitch=NERO_LEFT_BASE_RPY[1],
                yaw=NERO_LEFT_BASE_RPY[2],
            )
        elif arm == "right_arm":
            self._task_name = "cartesian_ik_right_arm"
            bp = base_pose(
                x=NERO_BASE_X,
                y=NERO_RIGHT_BASE_Y,
                z=NERO_BASE_Z,
                roll=NERO_RIGHT_BASE_RPY[0],
                pitch=NERO_RIGHT_BASE_RPY[1],
                yaw=NERO_RIGHT_BASE_RPY[2],
            )
        else:
            raise ValueError(f"arm must be 'left_arm' or 'right_arm', got {arm!r}")

        if self.config.pattern not in _PATTERNS:
            raise ValueError(
                f"pattern must be one of {sorted(_PATTERNS)}, got {self.config.pattern!r}"
            )
        if self.config.axis not in _AXES:
            raise ValueError(f"axis must be one of {sorted(_AXES)}, got {self.config.axis!r}")
        if self.config.plane not in _PLANES:
            raise ValueError(f"plane must be one of {sorted(_PLANES)}, got {self.config.plane!r}")

        self._t_world_base = pose_to_matrix(bp)
        self._t_base_world = np.linalg.inv(self._t_world_base)
        self._joint_names = [f"{arm}/joint{i}" for i in range(1, NERO_DOF + 1)]
        self._ik = PinocchioIK.from_model_path(NERO_FK_MODEL, NERO_EE_JOINT_ID)

        # Mutable runtime state (guarded by _lock).
        self._center: np.ndarray | None = None  # world-frame [x,y,z]
        self._orientation: Quaternion | None = None  # held world orientation
        self._latest_q: np.ndarray | None = None
        self._pattern = self.config.pattern
        self._amplitude = float(self.config.amplitude)
        self._radius = float(self.config.radius)
        self._period = max(1e-3, float(self.config.period))
        self._axis = _AXES[self.config.axis]
        self._plane = _PLANES[self.config.plane]
        self._active = bool(self.config.active)

    # ------------------------------------------------------------------ lifecycle
    @rpc
    def start(self) -> None:
        super().start()
        self._stop.clear()
        self.coordinator_joint_state.subscribe(self._on_joint_state)
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info("NeroPatternPlannerModule started", arm=self.config.arm)

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    # ------------------------------------------------------------------ reference centre (RPC)
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
    def get_center(self) -> list[float]:
        """Return the current world-frame reference centre (or [] if unseeded)."""
        with self._lock:
            return [] if self._center is None else self._center.tolist()

    # ------------------------------------------------------------------ pattern control (RPC)
    @rpc
    def set_pattern(self, name: str) -> str:
        """Switch the active pattern ("hold", "line", or "circle")."""
        if name not in _PATTERNS:
            return f"unknown pattern {name!r}; choose from {sorted(_PATTERNS)}"
        with self._lock:
            self._pattern = name
        return f"pattern set to {name!r}"

    @rpc
    def list_patterns(self) -> list[str]:
        """Return the available pattern names."""
        return sorted(_PATTERNS)

    @rpc
    def set_amplitude(self, meters: float) -> str:
        """Set the line half-stroke amplitude in metres."""
        with self._lock:
            self._amplitude = float(meters)
        return f"amplitude set to {self._amplitude:.3f}m"

    @rpc
    def set_radius(self, meters: float) -> str:
        """Set the circle radius in metres."""
        with self._lock:
            self._radius = float(meters)
        return f"radius set to {self._radius:.3f}m"

    @rpc
    def set_axis(self, axis: str) -> str:
        """Set the world axis the line oscillates along ("x"/"y"/"z")."""
        if axis not in _AXES:
            return f"axis must be one of {sorted(_AXES)}, got {axis!r}"
        with self._lock:
            self._axis = _AXES[axis]
        return f"axis set to {axis!r}"

    @rpc
    def set_plane(self, plane: str) -> str:
        """Set the world plane the circle traces ("xy"/"xz"/"yz")."""
        if plane not in _PLANES:
            return f"plane must be one of {sorted(_PLANES)}, got {plane!r}"
        with self._lock:
            self._plane = _PLANES[plane]
        return f"plane set to {plane!r}"

    @rpc
    def set_period(self, seconds: float) -> str:
        """Set the pattern cycle period in seconds (higher = slower)."""
        with self._lock:
            self._period = max(1e-3, float(seconds))
        return f"period set to {self._period:.3f}s"

    @rpc
    def set_active(self, on: bool) -> str:
        """Enable/disable streaming. When off, the task times out and holds."""
        with self._lock:
            self._active = bool(on)
        return f"active={self._active}"

    @rpc
    def get_state(self) -> str:
        """Return a summary of the current pattern, shape params, and centre."""
        with self._lock:
            center = "unseeded" if self._center is None else np.round(self._center, 3).tolist()
            return (
                f"pattern={self._pattern} active={self._active} period={self._period:.3f}s "
                f"amplitude={self._amplitude:.3f}m radius={self._radius:.3f}m center={center}"
            )

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

    def _stream_loop(self) -> None:
        # Drift-compensated fixed-rate scheduler: schedule each publish at an
        # absolute deadline (next_t += dt) instead of sleeping a fixed dt after
        # variable per-iteration work. This keeps the sample interval uniform, so
        # the streamed velocity is smooth (plain time.sleep(dt) accumulates OS
        # scheduling jitter -> uneven spacing -> visible chop, worst at speed).
        dt = 1.0 / max(1e-3, self.config.rate_hz)
        t0 = time.perf_counter()
        next_t = t0
        while not self._stop.is_set():
            now = time.perf_counter()
            cmd: PoseStamped | None = None
            with self._lock:
                if self._center is None:
                    self._try_seed()
                if self._active and self._center is not None and self._orientation is not None:
                    phase = 2.0 * np.pi * ((now - t0) / self._period)
                    params = PatternParams(
                        amplitude=self._amplitude,
                        radius=self._radius,
                        axis=self._axis,
                        plane=self._plane,
                    )
                    offset = _PATTERNS[self._pattern](phase, params)
                    world_pos = self._center + offset
                    cmd = self._to_base_command(world_pos, self._orientation)
            if cmd is not None:
                self.coordinator_cartesian_command.publish(cmd)
            next_t += dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind schedule (a slow tick); resync to now so we don't
                # burst-publish to "catch up".
                next_t = time.perf_counter()

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
