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

"""Meta Quest 3 dual-arm teleop blueprints for the AgileX NERO.

Lives in the NERO package (not dimos.teleop.quest.blueprints) on purpose: that
module imports GO2Connection at import time, which drags in the unitree WebRTC
dependency that the manipulation-only environment used by the ADVX massage app
does not install. Keeping the NERO Quest blueprints here lets them import and
run under ``--extra manipulation`` with no unitree deps.

Wiring
------
Quest browser --WebSocket--> ArmTeleopModule (embedded HTTPS :8443)
    left_controller_output  (X hold to engage) -> teleop_left_arm
    right_controller_output (A hold to engage) -> teleop_right_arm

The ArmTeleopModule stamps each hand's PoseStamped ``frame_id`` with the task
name; both hands remap onto the coordinator's single
``coordinator_cartesian_command`` input, which routes by ``frame_id``.
``teleop_buttons`` autoconnects by name so each per-arm TeleopIKTask sees
engage/disengage. The NERO planner module is composed in only for
VISUALIZATION (viser: both arms + body + measured table); it passively renders
the coordinator's joint-state stream and does not command the arms.

Blueprints
----------
- ``teleop-quest-nero-mock``: mock adapters, no hardware. Safe Quest bring-up
  and viser preview of the arms following the controllers.
- ``teleop-quest-nero``: real dual-arm over CAN. Only run with the NERO
  powered, homed, and the workspace clear.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.manipulators.common.blueprints import planner
from dimos.robot.manipulators.nero.blueprints.basic import (
    coordinator_nero_teleop_bimanual,
    coordinator_nero_teleop_mock,
)
from dimos.robot.manipulators.nero.blueprints.planner import (
    left_model as nero_left_model,
    nero_startup_obstacles,
    nero_visualization,
    right_model as nero_right_model,
)
from dimos.teleop.quest.quest_extensions import ArmTeleopModule

# Hand -> coordinator task name (must match _nero_teleop_task names in basic.py)
_NERO_TASK_NAMES = {"left": "teleop_left_arm", "right": "teleop_right_arm"}

_NERO_TELEOP_REMAPS = [
    (ArmTeleopModule, "left_controller_output", "coordinator_cartesian_command"),
    (ArmTeleopModule, "right_controller_output", "coordinator_cartesian_command"),
]


def _nero_viz():
    """NERO planner module used purely for viser visualization here."""
    return planner(
        robots=[nero_left_model, nero_right_model],
        visualization=nero_visualization,
        startup_obstacles=nero_startup_obstacles,
    )


# Mock (no hardware) — safe Quest bring-up + viser preview of arms following.
teleop_quest_nero_mock = autoconnect(
    ArmTeleopModule.blueprint(task_names=_NERO_TASK_NAMES),
    _nero_viz(),
    coordinator_nero_teleop_mock,
).remappings(_NERO_TELEOP_REMAPS)


# Real dual-arm over CAN. Only run with the NERO powered, homed, and clear.
teleop_quest_nero = autoconnect(
    ArmTeleopModule.blueprint(task_names=_NERO_TASK_NAMES),
    _nero_viz(),
    coordinator_nero_teleop_bimanual,
).remappings(_NERO_TELEOP_REMAPS)
