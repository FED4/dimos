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

"""Basic AgileX NERO coordinator blueprints."""

from __future__ import annotations

from dimos.control.coordinator import ControlCoordinator
from dimos.robot.manipulators.common.blueprints import trajectory_task
from dimos.robot.manipulators.nero.config import (
    NERO_LEFT_CAN,
    NERO_RIGHT_CAN,
    nero_hardware,
    nero_real_hardware,
)

mock_left = nero_hardware("left_arm")
mock_right = nero_hardware("right_arm")

coordinator_nero_mock = ControlCoordinator.blueprint(
    hardware=[mock_left, mock_right],
    tasks=[
        trajectory_task(mock_left),
        trajectory_task(mock_right),
    ],
)

left_hw = nero_real_hardware("left_arm", address=NERO_LEFT_CAN)
right_hw = nero_real_hardware("right_arm", address=NERO_RIGHT_CAN)

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
