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

"""AgileX NERO cartesian pattern planner blueprints.

Bundled blueprints that add ``NeroPatternPlannerModule`` to the SAME cartesian
coordinator + Viser visualization used by ``coordinator-nero-cartesian-mock`` /
``-left`` (reusing the ``_cartesian_coordinator`` / ``_viz`` helpers from
``cartesian.py``). Everything runs under one ``dimos run`` (DimOS allows one
coordinator per LCM bus), and the planner module continuously streams a movable,
switchable pattern to the coordinator's ``cartesian_ik`` task.

    dimos run nero-pattern-planner-mock     # sim (Viser only, no CAN)
    dimos run nero-pattern-planner-left     # real left arm (can0)

Adjust the reference centre / pattern at runtime with the separate RPC client
(a plain script, not a ``dimos run``):

    python -m dimos.robot.manipulators.nero.scripts.demo_pattern_control

The existing ``coordinator-nero-cartesian-*`` blueprints are left unchanged.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.manipulators.nero.blueprints.cartesian import (
    CARTESIAN_IK_LEFT_TASK,
    _cartesian_coordinator,
    _viz,
    left_hw,
    left_model,
    mock_left,
)
from dimos.robot.manipulators.nero.pattern_planner_module import NeroPatternPlannerModule

# Mock: no CAN / no hardware. Stream patterns and watch the arm track in Viser.
nero_pattern_planner_mock = autoconnect(
    _cartesian_coordinator(mock_left, CARTESIAN_IK_LEFT_TASK),
    NeroPatternPlannerModule.blueprint(arm="left_arm"),
    _viz(left_model),
)

# Real left arm (can0).
nero_pattern_planner_left = autoconnect(
    _cartesian_coordinator(left_hw, CARTESIAN_IK_LEFT_TASK),
    NeroPatternPlannerModule.blueprint(arm="left_arm"),
    _viz(left_model),
)
