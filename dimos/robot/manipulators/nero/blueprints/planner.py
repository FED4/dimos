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

"""AgileX NERO manipulation planning blueprints."""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.manipulators.common.blueprints import planner
from dimos.robot.manipulators.nero.blueprints.basic import (
    coordinator_nero_bimanual,
    coordinator_nero_mock,
)
from dimos.robot.manipulators.nero.config import (
    NERO_BASE_X,
    NERO_BASE_Z,
    NERO_LEFT_BASE_RPY,
    NERO_LEFT_BASE_Y,
    NERO_RIGHT_BASE_RPY,
    NERO_RIGHT_BASE_Y,
    nero_body_static_model,
    nero_default_table_obstacle,
    nero_model_config,
)

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

nero_mock_planner_coordinator = autoconnect(
    planner(
        robots=[left_model, right_model],
        visualization=nero_visualization,
        startup_obstacles=nero_startup_obstacles,
    ),
    coordinator_nero_mock,
)

nero_planner_coordinator = autoconnect(
    planner(
        robots=[left_model, right_model],
        visualization=nero_visualization,
        startup_obstacles=nero_startup_obstacles,
    ),
    coordinator_nero_bimanual,
)
