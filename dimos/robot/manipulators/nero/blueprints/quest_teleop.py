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
from dimos.robot.manipulators.nero.blueprints.cartesian import (
    coordinator_nero_cartesian_bimanual,
    coordinator_nero_cartesian_bimanual_mock,
    coordinator_nero_cartesian_left,
    coordinator_nero_cartesian_right,
)
from dimos.robot.manipulators.nero.quest_stream_module import NeroControllerStreamModule
from dimos.robot.manipulators.nero.quest_teleop_module import NeroBimanualQuestTeleopModule

# Teleop tuning shared by the mock and real blueprints so they never drift.
# Settled during the Step 3 mock bring-up (axes verified in Viser):
#   swap_controllers  left controller -> left arm, right -> right (sides matched)
#   *_axis_map        world offset <- signed controller-delta axis, per arm
#   position_scale    controller metres -> world metres (0.5 = half; calmer)
#   max_offset_m      max reach from the engage anchor (workspace guard)
_TELEOP_TUNING: dict = dict(
    motion_enabled=True,
    position_scale=0.5,
    max_offset_m=0.30,
    swap_controllers=True,
    left_axis_map=["-x", "-y", "z"],
    right_axis_map=["-x", "-y", "z"],
)


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
    NeroControllerStreamModule.blueprint(),
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
    NeroControllerStreamModule.blueprint(),
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
    NeroControllerStreamModule.blueprint(),
    _teleop_bridge(),
)

teleop_quest_nero_right = autoconnect(
    coordinator_nero_cartesian_right,
    NeroControllerStreamModule.blueprint(),
    _teleop_bridge(),
)
