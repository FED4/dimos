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

"""Raw Quest controller pose streamer for NERO teleop.

A thin subclass of ``QuestTeleopModule`` that publishes each controller's
*absolute* pose (already transformed into the robot frame by the base class via
``webxr_to_robot``) on ``left_controller_output`` / ``right_controller_output``
every control tick, regardless of engage state, and always publishes
``teleop_buttons`` (which carries the per-hand trigger).

This deliberately does NOT gate on engagement and does NOT compute deltas. All
engagement (trigger press), clutch/anchoring, and delta maths live downstream in
``NeroBimanualQuestTeleopModule`` so there is a single source of truth for when
the arm moves. The base ``QuestTeleopModule`` still owns the embedded HTTPS/
WebSocket server on ``:8443`` that the Quest browser connects to.
"""

from __future__ import annotations

import time

from dimos.core.core import rpc
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.teleop.quest.quest_teleop_module import QuestTeleopConfig, QuestTeleopModule
from dimos.teleop.quest.quest_types import Buttons, Hand, QuestControllerState
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Throttle for the human-readable telemetry log (seconds between lines).
_TELEMETRY_LOG_PERIOD_S = 0.5


class NeroControllerStreamModule(QuestTeleopModule):
    """Streams absolute robot-frame controller poses for both hands.

    Outputs (inherited from QuestTeleopModule):
        - left_controller_output: PoseStamped (absolute robot-frame pose)
        - right_controller_output: PoseStamped (absolute robot-frame pose)
        - teleop_buttons: Buttons (per-hand trigger/grip/face buttons)

    Also emits a throttled telemetry log (~2 Hz) of each hand's position and
    trigger so Step 1 is observable in ``dimos log -f`` without extra tooling.
    """

    config: QuestTeleopConfig

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._last_telemetry_log = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        logger.info("NeroControllerStreamModule started (absolute pose passthrough)")

    @rpc
    def stop(self) -> None:
        super().stop()

    def _handle_engage(self) -> None:
        """No-op: engagement is owned by the downstream teleop bridge (trigger).

        Overrides the base primary-button (X/A) engage so it doesn't emit
        confusing "LEFT engaged" logs or gate anything here.
        """
        return

    def _should_publish(self, hand: Hand) -> bool:
        """Always publish; engagement is decided downstream by the teleop bridge."""
        return True

    def _get_output_pose(self, hand: Hand) -> PoseStamped | None:
        """Return the latest absolute robot-frame controller pose (no delta)."""
        return self._current_poses.get(hand)

    def _publish_button_state(
        self,
        left: QuestControllerState | None,
        right: QuestControllerState | None,
    ) -> None:
        """Publish buttons WITH analog triggers packed, then log telemetry.

        The base QuestTeleopModule does not pack analog trigger bits (only
        ArmTeleopModule does), so we replicate that packing here — the teleop
        bridge reads ``left_trigger_analog`` / ``right_trigger_analog`` to decide
        engagement.
        """
        buttons = Buttons.from_controllers(left, right)
        buttons.pack_analog_triggers(
            left=left.trigger if left is not None else 0.0,
            right=right.trigger if right is not None else 0.0,
        )
        self.teleop_buttons.publish(buttons)

        now = time.perf_counter()
        if now - self._last_telemetry_log < _TELEMETRY_LOG_PERIOD_S:
            return
        self._last_telemetry_log = now
        logger.info(
            "quest telemetry",
            left=self._describe(Hand.LEFT, left),
            right=self._describe(Hand.RIGHT, right),
        )

    def _describe(self, hand: Hand, ctrl: QuestControllerState | None) -> str:
        pose = self._current_poses.get(hand)
        if pose is None or ctrl is None:
            return "no data"
        p = pose.position
        return (
            f"pos=({p.x:+.3f},{p.y:+.3f},{p.z:+.3f}) "
            f"trigger={ctrl.trigger:.2f} grip={ctrl.grip:.2f} primary={int(ctrl.primary)}"
        )
