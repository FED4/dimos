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
from dimos.robot.manipulators.nero.config import nero_model_config

left_model = nero_model_config("left_arm")
right_model = nero_model_config("right_arm")

nero_mock_planner_coordinator = autoconnect(
    planner(robots=[left_model, right_model]),
    coordinator_nero_mock,
)

nero_planner_coordinator = autoconnect(
    planner(robots=[left_model, right_model]),
    coordinator_nero_bimanual,
)
