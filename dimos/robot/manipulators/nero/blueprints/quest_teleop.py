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

"""Meta Quest teleoperation blueprints for the AgileX NERO arms.

Bimanual, trigger-to-engage teleop built on the healthy cartesian streaming
path: a ``NeroControllerStreamModule`` (Quest HTTPS/WS server on :8443) streams
absolute controller poses, and ``NeroBimanualQuestTeleopModule`` turns them into
world->base cartesian commands for the SAME ``coordinator-nero-cartesian-*``
coordinators used elsewhere (reused unchanged).

Blueprints (built incrementally):

    dimos run teleop-quest-nero-telemetry    # Step 1: stream + mock coordinator,
                                             #         controllers only, NO arm motion
    dimos run teleop-quest-nero-engage-mock  # Step 2: + trigger-engage bridge,
                                             #         logs FK anchors, still NO motion
    dimos run teleop-quest-nero-bimanual-mock  # Step 3: motion ENABLED, arms track
                                               #         the controllers in Viser (mock)
    dimos run teleop-quest-nero-bimanual       # Step 4: REAL arms (can0 + can1)
    dimos run teleop-quest-nero-left           # REAL left arm only  (can0)
    dimos run teleop-quest-nero-right          # REAL right arm only (can1)
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.robot.manipulators.common.blueprints import cartesian_ik_task, coordinator
from dimos.robot.manipulators.nero.blueprints.cartesian import (
    CARTESIAN_IK_LEFT_TASK,
    CARTESIAN_IK_RIGHT_TASK,
    coordinator_nero_cartesian_bimanual,
    coordinator_nero_cartesian_bimanual_mock,
    coordinator_nero_cartesian_left,
    coordinator_nero_cartesian_right,
    left_model,
    nero_startup_obstacles,
    nero_visualization,
    right_model,
)
from dimos.robot.manipulators.nero.config import (
    NERO_EE_JOINT_ID,
    NERO_FK_MODEL,
    NERO_LEFT_CAN,
    NERO_RIGHT_CAN,
    nero_real_hardware,
)
from dimos.robot.manipulators.nero.quest_stream_module import NeroControllerStreamModule
from dimos.robot.manipulators.nero.quest_teleop_module import NeroBimanualQuestTeleopModule

# Controller sampling rate for the Quest stream module. The default 50 Hz
# produced a zero-order-hold staircase against the ~100 Hz command loop (each
# sample held for two ticks -> step changes -> IK jumps -> visible jitter).
# 90 Hz matches the Quest 3 refresh and the client's ~80 Hz send cap.
_CONTROLLER_RATE_HZ = 90.0

# Teleop tuning shared by the mock and real blueprints so they never drift.
# Sides/axes settled during the Step 3 mock bring-up (verified in Viser):
#   swap_controllers  which controller drives which arm
#   *_axis_map        world offset <- signed controller-delta axis, per arm
#                     (also conjugated onto orientation, so rotation matches)
#   position_scale    controller metres -> world metres (0.5 = half; calmer)
#   max_offset_m      max reach from the engage anchor (workspace guard)
#
# Smoothness / robustness (see quest_teleop_module docstring for why):
#   track_orientation      follow controller rotation. Holding a fixed
#                          orientation over-constrains the arm and is a common
#                          cause of unreachable targets and wedged poses.
#   rotation_scale         fraction of controller rotation applied (0.5 = calm)
#   smoothing_tau_s        low-pass time constant on the target
#   max_speed_mps          target speed limit -- keeps IK inside the task's
#                          per-tick joint-delta clamp (which otherwise REJECTS
#                          the solution and emits nothing -> stall/lurch)
#   max_angular_speed_dps  target angular speed limit
#   max_tracking_error_m   the leash: target may never sit further than this
#                          from the measured EE, so a saturated or unreachable
#                          arm recovers instead of staying wedged
#   deadband_m             ignore sub-mm controller dither
_TELEOP_TUNING: dict = dict(
    motion_enabled=True,
    position_scale=0.5,
    max_offset_m=0.30,
    swap_controllers=True,
    left_axis_map=["-x", "-y", "z"],
    right_axis_map=["-x", "-y", "z"],
    track_orientation=True,
    rotation_scale=0.5,
    smoothing_tau_s=0.08,
    max_speed_mps=0.35,
    max_angular_speed_dps=120.0,
    max_tracking_error_m=0.12,
    deadband_m=0.001,
)


def _controller_stream():  # type: ignore[no-untyped-def]
    """Quest controller stream module at the tuned sample rate."""
    return NeroControllerStreamModule.blueprint(control_loop_hz=_CONTROLLER_RATE_HZ)


def _teleop_bridge():  # type: ignore[no-untyped-def]
    """The bimanual Quest->NERO bridge with the settled tuning."""
    return NeroBimanualQuestTeleopModule.blueprint(**_TELEOP_TUNING)

# ---------------------------------------------------------------------------
# Step 1 -- telemetry only. Confirms the Quest connects at :8443 and both
# controllers' absolute robot-frame poses + trigger states are published. No
# teleop bridge yet, so the mock arms do not move.
# ---------------------------------------------------------------------------
teleop_quest_nero_telemetry = autoconnect(
    coordinator_nero_cartesian_bimanual_mock,
    NeroControllerStreamModule.blueprint(),
)

# ---------------------------------------------------------------------------
# Step 2 -- engage bridge, motion DISABLED. Adds NeroBimanualQuestTeleopModule
# so trigger rising/falling edges are detected and each arm's world-frame FK
# anchor is logged on engage, but no cartesian command is published, so the
# mock arms still do not move. Verifies FK seeding + trigger edges.
# ---------------------------------------------------------------------------
teleop_quest_nero_engage_mock = autoconnect(
    coordinator_nero_cartesian_bimanual_mock,
    NeroControllerStreamModule.blueprint(),
    NeroBimanualQuestTeleopModule.blueprint(motion_enabled=False),
)

# ---------------------------------------------------------------------------
# Step 3 -- motion ENABLED on the mock bimanual coordinator. Each arm tracks its
# controller (position-only) while its trigger is held, holds on release. Watch
# both mock arms move in Viser. Tune position_scale / max_offset_m here and
# check that controller axes map intuitively to base axes before real hardware.
# ---------------------------------------------------------------------------
teleop_quest_nero_bimanual_mock = autoconnect(
    coordinator_nero_cartesian_bimanual_mock,
    _controller_stream(),
    _teleop_bridge(),
)

# ---------------------------------------------------------------------------
# Step 4 -- REAL bimanual teleop on hardware (left=can0, right=can1). Identical
# wiring and tuning to the mock above, but driving the real cartesian_ik
# coordinator instead of the mock one. The same Viser twin still renders both
# arms live. Both CAN buses must be up: a connect failure on either prevents the
# coordinator from starting (inherent to coordinator_nero_cartesian_bimanual).
#
# SAFETY: this path has NO collision / joint-limit avoidance beyond the
# cartesian_ik per-tick joint-delta clamp. max_offset_m is the workspace guard
# and trigger-release is the deadman. Keep a hand on the e-stop for first runs;
# lower position_scale in _TELEOP_TUNING for calmer motion.
# ---------------------------------------------------------------------------
teleop_quest_nero_bimanual = autoconnect(
    coordinator_nero_cartesian_bimanual,
    _controller_stream(),
    _teleop_bridge(),
)

# ---------------------------------------------------------------------------
# Single-arm REAL teleop. Same bridge and tuning, but a fault-isolated per-arm
# coordinator so you can bring up (or debug) one arm without the other's CAN bus.
# The bridge still listens to both controllers; the arm whose coordinator is not
# running simply never receives joint state, so it never engages. Use these when
# only one arm is powered/connected, or to isolate a CAN fault.
#
# NOTE: the shared tuning sets swap_controllers=True, so the controller that
# drives each arm is CROSSED:
#   teleop-quest-nero-left  (can0, left arm)  <- RIGHT controller trigger
#   teleop-quest-nero-right (can1, right arm) <- LEFT controller trigger
# If a single arm feels backwards on its own coordinator, flip swap_controllers
# in _TELEOP_TUNING.
# ---------------------------------------------------------------------------
teleop_quest_nero_left = autoconnect(
    coordinator_nero_cartesian_left,
    _controller_stream(),
    _teleop_bridge(),
)

teleop_quest_nero_right = autoconnect(
    coordinator_nero_cartesian_right,
    _controller_stream(),
    _teleop_bridge(),
)


# ---------------------------------------------------------------------------
# move_j teleop -- driver-planned motion instead of raw CPV setpoints.
#
# CPV streams position-velocity setpoints straight to each joint, so any
# roughness in the IK solution shows up directly as jitter/jumps at the joint.
# The "move_j" servo backend instead hands each streamed target to the driver as
# a planned joint move: the driver interpolates toward it and a new target
# supersedes the previous move. That trades latency for much smoother motion,
# and tolerates a lower command rate.
#
# Differences from the CPV blueprints above:
#   * hardware built with servo_backend="move_j"
#   * command_rate_hz throttled (a planned move should not be re-issued at
#     100 Hz or the driver re-plans constantly); still far above the
#     cartesian_ik task's 0.5 s timeout
#   * lighter bridge smoothing, since the driver now does the interpolation
# ---------------------------------------------------------------------------

_MOVE_J_COMMAND_RATE_HZ = 25.0

_TELEOP_TUNING_MOVE_J: dict = dict(
    _TELEOP_TUNING,
    command_rate_hz=_MOVE_J_COMMAND_RATE_HZ,
    # The driver interpolates now, so less bridge-side filtering is needed.
    smoothing_tau_s=0.05,
    # Planned moves lag, so allow a longer leash before clamping the target.
    max_tracking_error_m=0.18,
)

left_hw_move_j = nero_real_hardware(
    "left_arm", address=NERO_LEFT_CAN, servo_backend="move_j"
)
right_hw_move_j = nero_real_hardware(
    "right_arm", address=NERO_RIGHT_CAN, servo_backend="move_j"
)


def _teleop_bridge_move_j():  # type: ignore[no-untyped-def]
    """Bimanual bridge tuned for the driver-planned move_j backend."""
    return NeroBimanualQuestTeleopModule.blueprint(**_TELEOP_TUNING_MOVE_J)


def _viz_move_j(*models):  # type: ignore[no-untyped-def]
    """Viser twin for the move_j blueprints (mirrors cartesian._viz)."""
    return ManipulationModule.blueprint(
        robots=list(models),
        visualization=nero_visualization,
        startup_obstacles=nero_startup_obstacles,
    )


def _cartesian_coordinator_move_j(hardware, task_name):  # type: ignore[no-untyped-def]
    """One cartesian_ik task at 100 Hz against move_j hardware."""
    return coordinator(
        hardware=[hardware],
        tasks=[
            cartesian_ik_task(
                hardware,
                model_path=NERO_FK_MODEL,
                ee_joint_id=NERO_EE_JOINT_ID,
                name=task_name,
            )
        ],
        tick_rate=100.0,
    )


# Real bimanual, driver-planned move_j (left=can0, right=can1).
teleop_quest_nero_bimanual_move_j = autoconnect(
    coordinator(
        hardware=[left_hw_move_j, right_hw_move_j],
        tasks=[
            cartesian_ik_task(
                left_hw_move_j,
                model_path=NERO_FK_MODEL,
                ee_joint_id=NERO_EE_JOINT_ID,
                name=CARTESIAN_IK_LEFT_TASK,
            ),
            cartesian_ik_task(
                right_hw_move_j,
                model_path=NERO_FK_MODEL,
                ee_joint_id=NERO_EE_JOINT_ID,
                name=CARTESIAN_IK_RIGHT_TASK,
            ),
        ],
        tick_rate=100.0,
    ),
    _viz_move_j(left_model, right_model),
    _controller_stream(),
    _teleop_bridge_move_j(),
)

# Real single-arm, driver-planned move_j (fault-isolated per CAN bus).
teleop_quest_nero_left_move_j = autoconnect(
    _cartesian_coordinator_move_j(left_hw_move_j, CARTESIAN_IK_LEFT_TASK),
    _viz_move_j(left_model),
    _controller_stream(),
    _teleop_bridge_move_j(),
)

teleop_quest_nero_right_move_j = autoconnect(
    _cartesian_coordinator_move_j(right_hw_move_j, CARTESIAN_IK_RIGHT_TASK),
    _viz_move_j(right_model),
    _controller_stream(),
    _teleop_bridge_move_j(),
)

# Mock counterpart: exercises the throttled command rate + lighter smoothing
# without hardware. The mock adapter ignores the servo backend, so this validates
# the tuning and wiring only, not the driver-side planning behaviour.
teleop_quest_nero_bimanual_move_j_mock = autoconnect(
    coordinator_nero_cartesian_bimanual_mock,
    _controller_stream(),
    _teleop_bridge_move_j(),
)
