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

"""Runtime controller for the NERO poke pattern module (RPC client).

Drives NeroCartesianPatternModule at runtime: move the reference centre and
reshape the motion while it streams. These are the same @rpc methods that will
later be exposed as @skill for natural-language control.

Usage:
    # Start a poke blueprint in another terminal first:
    #   dimos run coordinator-nero-poke-mock      # simulate (Viser)
    #   dimos run coordinator-nero-poke-left      # real left arm
    #
    # Then run this interactive client:
    python -i -m dimos.robot.manipulators.nero.scripts.demo_poke_control

Available functions:
    center()                 current world-frame reference centre
    set_center(x, y, z)      move the poke centre to an absolute world point
    nudge(dx, dy, dz)        shift the centre by a world delta
    set_period(s)            seconds per poke cycle (higher = slower)
    set_amplitude(m)         poke half-stroke (metres)
    set_sweep_speed(mps)     centre auto-drift along world X (0 = stop)
    set_active(on)           enable/disable streaming
"""

from __future__ import annotations

from dimos.core.rpc_client import RPCClient
from dimos.robot.manipulators.nero.cartesian_pattern_module import NeroCartesianPatternModule

_client = RPCClient(None, NeroCartesianPatternModule)


def center() -> list[float]:
    """Return the current world-frame reference centre."""
    return _client.get_center()


def set_center(x: float, y: float, z: float) -> str:
    """Move the poke centre to an absolute world position (metres)."""
    return _client.set_center(x, y, z)


def nudge(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    """Shift the poke centre by a world-frame delta (metres)."""
    return _client.nudge(dx, dy, dz)


def set_period(seconds: float) -> str:
    """Set the poke cycle period in seconds."""
    return _client.set_period(seconds)


def set_amplitude(meters: float) -> str:
    """Set the poke half-stroke amplitude in metres."""
    return _client.set_amplitude(meters)


def set_sweep_speed(mps: float) -> str:
    """Set the centre auto-drift speed along world X (m/s, 0 = stop)."""
    return _client.set_sweep_speed(mps)


def set_active(on: bool) -> str:
    """Enable or disable streaming."""
    return _client.set_active(on)


def stop() -> None:
    """Stop the RPC client."""
    _client.stop_rpc_client()


if __name__ == "__main__":
    print("NERO poke controller ready. Try:")
    print("  center(); set_center(0.4, 0.1, 0.35); nudge(dz=0.05); set_period(3); set_sweep_speed(0)")
