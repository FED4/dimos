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

"""Basic AgileX NERO coordinator blueprints.

Servo blueprints (coordinator_nero_servo_*)
-------------------------------------------
Use the ``"servo"`` task type which routes streaming ``JointState`` messages
published on ``/joint_command`` directly to hardware via CPV mode
(``move_cpv_pos`` per joint) at 100 Hz.  These are the entry point for:

  - CPV 100 Hz streaming validation (demo_cpv_servo_test.py)
  - Teleoperation
  - Learned-policy joint-space control

The coordinator tick rate is set to 100 Hz to match the target control
frequency.  The servo task has a 0.5 s command timeout: if no new
``JointState`` is received within that window the task goes inactive and the
arm holds its last commanded position.

Dual-arm planning note
----------------------
Each arm is planned independently using the single-arm URDF placed at its
physical base-pose offset (same pattern as ``dual_xarm6_planner``).  The
combined wbcd xacro is used for visualisation / simulation only — the
Pinocchio IK solver in CartesianIKTask requires a single kinematic chain and
therefore receives per-arm models.
"""

from __future__ import annotations

from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.robot.manipulators.common.blueprints import trajectory_task
from dimos.robot.manipulators.nero.config import (
    NERO_LEFT_CAN,
    NERO_RIGHT_CAN,
    NERO_LEFT_BASE_Y,
    NERO_RIGHT_BASE_Y,
    NERO_BASE_Z,
    nero_hardware,
    nero_real_hardware,
)

# ---------------------------------------------------------------------------
# Shared hardware components
# ---------------------------------------------------------------------------

mock_left = nero_hardware("left_arm")
mock_right = nero_hardware("right_arm")

left_hw = nero_real_hardware("left_arm", address=NERO_LEFT_CAN)
right_hw = nero_real_hardware("right_arm", address=NERO_RIGHT_CAN)

# ---------------------------------------------------------------------------
# Trajectory-task coordinators (planned motion, POSITION mode / move_j)
# ---------------------------------------------------------------------------

coordinator_nero_mock = ControlCoordinator.blueprint(
    hardware=[mock_left, mock_right],
    tasks=[
        trajectory_task(mock_left),
        trajectory_task(mock_right),
    ],
)

coordinator_nero_left = ControlCoordinator.blueprint(
    hardware=[left_hw],
    tasks=[trajectory_task(left_hw)],
)

coordinator_nero_right = ControlCoordinator.blueprint(
    hardware=[right_hw],
    tasks=[trajectory_task(right_hw)],
)

coordinator_nero_bimanual = ControlCoordinator.blueprint(
    hardware=[left_hw, right_hw],
    tasks=[
        trajectory_task(left_hw),
        trajectory_task(right_hw),
    ],
)

# ---------------------------------------------------------------------------
# Servo-task coordinators (100 Hz CPV streaming, SERVO_POSITION mode)
#
# Publish JointState to /joint_command to drive these coordinators.
# Joint names must match the hardware component's joint list, e.g.:
#   left_arm/joint1 .. left_arm/joint7
#   right_arm/joint1 .. right_arm/joint7
# ---------------------------------------------------------------------------

_SERVO_TICK_RATE = 100.0  # Hz — matches CPV target control frequency
_SERVO_TIMEOUT = 0.5      # seconds — arm holds position if stream goes silent

coordinator_nero_servo_left = ControlCoordinator.blueprint(
    tick_rate=_SERVO_TICK_RATE,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[left_hw],
    tasks=[
        TaskConfig(
            name="servo_left_arm",
            type="servo",
            joint_names=left_hw.joints,
            priority=10,
            params={"timeout": _SERVO_TIMEOUT},
        ),
    ],
)

coordinator_nero_servo_right = ControlCoordinator.blueprint(
    tick_rate=_SERVO_TICK_RATE,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[right_hw],
    tasks=[
        TaskConfig(
            name="servo_right_arm",
            type="servo",
            joint_names=right_hw.joints,
            priority=10,
            params={"timeout": _SERVO_TIMEOUT},
        ),
    ],
)

coordinator_nero_servo_bimanual = ControlCoordinator.blueprint(
    tick_rate=_SERVO_TICK_RATE,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[left_hw, right_hw],
    tasks=[
        TaskConfig(
            name="servo_left_arm",
            type="servo",
            joint_names=left_hw.joints,
            priority=10,
            params={"timeout": _SERVO_TIMEOUT},
        ),
        TaskConfig(
            name="servo_right_arm",
            type="servo",
            joint_names=right_hw.joints,
            priority=10,
            params={"timeout": _SERVO_TIMEOUT},
        ),
    ],
)
