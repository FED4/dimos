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

"""AgileX NERO continuous cartesian IK blueprints.

These blueprints run the streaming ``cartesian_ik`` control task: a target
end-effector pose is streamed on ``/coordinator_cartesian_command`` (PoseStamped,
frame_id == task name) and the task solves inverse kinematics internally with
Pinocchio each tick, emitting SERVO_POSITION joint commands that the NERO adapter
drives over CPV (``move_cpv_pos``). This is reactive control (no plan, no
freshness check); collision avoidance is NOT performed by this path.

Composition mirrors ``keyboard_teleop_piper`` /
``coordinator_cartesian_ik_piper``: a ``ControlCoordinator`` running the
``cartesian_ik`` task, plus a ``ManipulationModule`` used only for Viser
visualization (it subscribes to the coordinator's published joint state and
renders the arm live, so motion can be simulated on the mock before touching
real hardware).

Per-arm coordinators are used (not one bimanual coordinator) for fault isolation:
a CAN/connect failure on one arm does not take down the other. Because the IK
path does no collision checking, single vs bimanual grouping has no safety
effect here.

Streamed targets are expressed in the WORLD frame by the publisher (see
scripts/demo_cartesian_stream.py) and transformed into each arm's base frame
before publishing, because the control-side Pinocchio model has its base_link at
the origin.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.robot.manipulators.common.blueprints import cartesian_ik_task, coordinator
from dimos.robot.manipulators.nero.cartesian_pattern_module import NeroCartesianPatternModule
from dimos.robot.manipulators.nero.config import (
    NERO_BASE_X,
    NERO_BASE_Z,
    NERO_EE_JOINT_ID,
    NERO_FK_MODEL,
    NERO_LEFT_BASE_RPY,
    NERO_LEFT_BASE_Y,
    NERO_LEFT_CAN,
    NERO_RIGHT_BASE_RPY,
    NERO_RIGHT_BASE_Y,
    NERO_RIGHT_CAN,
    nero_body_static_model,
    nero_default_table_obstacle,
    nero_hardware,
    nero_model_config,
    nero_real_hardware,
)

# Task names double as the routing key: the publisher sets PoseStamped.frame_id
# to one of these so the coordinator delivers it to the matching arm's task.
CARTESIAN_IK_LEFT_TASK = "cartesian_ik_left_arm"
CARTESIAN_IK_RIGHT_TASK = "cartesian_ik_right_arm"

# Per-arm planning/visualization models (same placement as the planner blueprint).
left_model = nero_model_config(
    "left_arm",
    x_offset=NERO_BASE_X,
    y_offset=NERO_LEFT_BASE_Y,
    z_offset=NERO_BASE_Z,
    roll=NERO_LEFT_BASE_RPY[0],
    pitch=NERO_LEFT_BASE_RPY[1],
    yaw=NERO_LEFT_BASE_RPY[2],
)
right_model = nero_model_config(
    "right_arm",
    x_offset=NERO_BASE_X,
    y_offset=NERO_RIGHT_BASE_Y,
    z_offset=NERO_BASE_Z,
    roll=NERO_RIGHT_BASE_RPY[0],
    pitch=NERO_RIGHT_BASE_RPY[1],
    yaw=NERO_RIGHT_BASE_RPY[2],
)

nero_visualization = {
    "backend": "viser",
    "robot_display_mode": "collision",
    "static_models": [nero_body_static_model()],
}
nero_startup_obstacles = [nero_default_table_obstacle()]

# Hardware components (7 arm joints, no gripper -> matches the FK model DOF).
mock_left = nero_hardware("left_arm")
mock_right = nero_hardware("right_arm")
left_hw = nero_real_hardware("left_arm", address=NERO_LEFT_CAN)
right_hw = nero_real_hardware("right_arm", address=NERO_RIGHT_CAN)


def _cartesian_coordinator(hardware, task_name):  # type: ignore[no-untyped-def]
    """ControlCoordinator running one cartesian_ik task at 100 Hz (CPV)."""
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


def _viz(*models):  # type: ignore[no-untyped-def]
    """ManipulationModule used only to visualize the live arm(s) in Viser."""
    return ManipulationModule.blueprint(
        robots=list(models),
        visualization=nero_visualization,
        startup_obstacles=nero_startup_obstacles,
    )


# Mock: no CAN / no hardware. Stream targets and watch the arm track in Viser.
coordinator_nero_cartesian_mock = autoconnect(
    _cartesian_coordinator(mock_left, CARTESIAN_IK_LEFT_TASK),
    _viz(left_model),
)

# Real single-arm coordinators (fault-isolated).
coordinator_nero_cartesian_left = autoconnect(
    _cartesian_coordinator(left_hw, CARTESIAN_IK_LEFT_TASK),
    _viz(left_model),
)

coordinator_nero_cartesian_right = autoconnect(
    _cartesian_coordinator(right_hw, CARTESIAN_IK_RIGHT_TASK),
    _viz(right_model),
)

# Bimanual: both arms in one coordinator, each driven by its own cartesian_ik
# task and its own command stream (routed by frame_id == task name). Publish to
# both task names to control both arms simultaneously (e.g. two streamer
# instances, one per --arm). Note: a single coordinator means a connect failure
# on either CAN bus prevents the whole stack from starting; the per-arm
# coordinators above remain available for isolated bring-up/debugging.
coordinator_nero_cartesian_bimanual = autoconnect(
    coordinator(
        hardware=[left_hw, right_hw],
        tasks=[
            cartesian_ik_task(
                left_hw,
                model_path=NERO_FK_MODEL,
                ee_joint_id=NERO_EE_JOINT_ID,
                name=CARTESIAN_IK_LEFT_TASK,
            ),
            cartesian_ik_task(
                right_hw,
                model_path=NERO_FK_MODEL,
                ee_joint_id=NERO_EE_JOINT_ID,
                name=CARTESIAN_IK_RIGHT_TASK,
            ),
        ],
        tick_rate=100.0,
    ),
    _viz(left_model, right_model),
)

# ---------------------------------------------------------------------------
# Poke blueprints: coordinator + a runtime-controllable pattern generator module
# (NeroCartesianPatternModule) that streams a movable "poke" (up/down oscillation)
# whose centre auto-sweeps along world X. The centre, period, amplitude and sweep
# are adjustable at runtime via the module's @rpc methods (see
# scripts/demo_poke_control.py), and are the hook for skills/natural-language
# control later. Do NOT also run demo_cartesian_stream against these blueprints
# (two publishers on the same command stream would fight).
# ---------------------------------------------------------------------------
coordinator_nero_poke_mock = autoconnect(
    _cartesian_coordinator(mock_left, CARTESIAN_IK_LEFT_TASK),
    NeroCartesianPatternModule.blueprint(arm="left_arm"),
    _viz(left_model),
)

coordinator_nero_poke_left = autoconnect(
    _cartesian_coordinator(left_hw, CARTESIAN_IK_LEFT_TASK),
    NeroCartesianPatternModule.blueprint(arm="left_arm"),
    _viz(left_model),
)

coordinator_nero_poke_right = autoconnect(
    _cartesian_coordinator(right_hw, CARTESIAN_IK_RIGHT_TASK),
    NeroCartesianPatternModule.blueprint(arm="right_arm"),
    _viz(right_model),
)

# Mock bimanual for simulation (both arms, no CAN).
coordinator_nero_cartesian_bimanual_mock = autoconnect(
    coordinator(
        hardware=[mock_left, mock_right],
        tasks=[
            cartesian_ik_task(
                mock_left,
                model_path=NERO_FK_MODEL,
                ee_joint_id=NERO_EE_JOINT_ID,
                name=CARTESIAN_IK_LEFT_TASK,
            ),
            cartesian_ik_task(
                mock_right,
                model_path=NERO_FK_MODEL,
                ee_joint_id=NERO_EE_JOINT_ID,
                name=CARTESIAN_IK_RIGHT_TASK,
            ),
        ],
        tick_rate=100.0,
    ),
    _viz(left_model, right_model),
)
