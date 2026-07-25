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

"""Runtime controller for the NERO pattern planner module (RPC client).

Drives NeroPatternPlannerModule at runtime: move the reference centre and switch
the pattern while it streams. These are the same @rpc methods that will later be
exposed as @skill for natural-language control.

Two ways to use it:

1. Interactive arrow-key jog loop (default) -- continuously nudge the reference
   centre in the world X/Z plane with the arrow keys and cycle patterns:

       # Start the coordinator + planner in two other terminals first:
       #   dimos run coordinator-nero-cartesian-mock    # sim (Viser)
       #   dimos run nero-pattern-planner-left          # the planner module
       python -m dimos.robot.manipulators.nero.scripts.demo_pattern_control

   Controls:
       Up / Down     nudge centre +Z / -Z (up / down)
       Right / Left  nudge centre +X / -X (forward / back)
       w / s         nudge centre +Y / -Y (left / right)
       + / -         grow / shrink the nudge step
       p             cycle pattern (hold -> line -> circle)
       SPACE         pause / resume streaming (set_active)
       q / ESC       quit

2. Manual REPL, for scripted/one-off calls:

       python -i -m dimos.robot.manipulators.nero.scripts.demo_pattern_control
       >>> center(); set_center(0.4, 0.1, 0.35); set_pattern("circle"); set_active(True)

Available functions:
    center()                 current world-frame reference centre
    set_center(x, y, z)      move the centre to an absolute world point
    nudge(dx, dy, dz)        shift the centre by a world delta
    set_pattern(name)        switch pattern ("hold" / "line" / "circle")
    list_patterns()          available pattern names
    set_amplitude(m)         line half-stroke (metres)
    set_radius(m)            circle radius (metres)
    set_axis(a)              line world axis ("x"/"y"/"z")
    set_plane(p)             circle world plane ("xy"/"xz"/"yz")
    set_period(s)            seconds per pattern cycle
    set_active(on)           enable/disable streaming
    get_state()              summary of pattern + params + centre
    jog()                    start the interactive arrow-key jog loop
"""

from __future__ import annotations

import curses
import traceback
from typing import Any

from dimos.core.rpc_client import RPCClient
from dimos.robot.manipulators.nero.pattern_planner_module import NeroPatternPlannerModule

_client = RPCClient(None, NeroPatternPlannerModule)

# Jog defaults.
_DEFAULT_STEP = 0.01  # metres per key press
_MIN_STEP = 0.001
_MAX_STEP = 0.10
_PATTERN_CYCLE = ["hold", "line", "circle"]


def center() -> list[float]:
    """Return the current world-frame reference centre."""
    return _client.get_center()


def set_center(x: float, y: float, z: float) -> str:
    """Move the centre to an absolute world position (metres)."""
    return _client.set_center(x, y, z)


def nudge(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    """Shift the centre by a world-frame delta (metres)."""
    return _client.nudge(dx, dy, dz)


def set_pattern(name: str) -> str:
    """Switch the active pattern ("hold" / "line" / "circle")."""
    return _client.set_pattern(name)


def list_patterns() -> list[str]:
    """Return the available pattern names."""
    return _client.list_patterns()


def set_amplitude(meters: float) -> str:
    """Set the line half-stroke amplitude in metres."""
    return _client.set_amplitude(meters)


def set_radius(meters: float) -> str:
    """Set the circle radius in metres."""
    return _client.set_radius(meters)


def set_axis(axis: str) -> str:
    """Set the line world axis ("x"/"y"/"z")."""
    return _client.set_axis(axis)


def set_plane(plane: str) -> str:
    """Set the circle world plane ("xy"/"xz"/"yz")."""
    return _client.set_plane(plane)


def set_period(seconds: float) -> str:
    """Set the pattern cycle period in seconds."""
    return _client.set_period(seconds)


def set_active(on: bool) -> str:
    """Enable or disable streaming."""
    return _client.set_active(on)


def get_state() -> str:
    """Return a summary of the current pattern, shape params, and centre."""
    return _client.get_state()


def stop() -> None:
    """Stop the RPC client."""
    _client.stop_rpc_client()


def _fmt_center(c: list[float]) -> str:
    """Format the world-frame centre for display."""
    if not c:
        return "(not seeded yet -- is a coordinator running?)"
    return f"x={c[0]:+.3f}  y={c[1]:+.3f}  z={c[2]:+.3f}  (metres, world)"


def _draw_ui(stdscr: Any, step: float, pattern: str, active: bool, status: str) -> None:
    """Render the jog control UI."""
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    title = "NERO pattern planner jog"
    stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_BOLD)

    lines = [
        "",
        "Nudge the reference centre (world frame):",
        "  Up / Down     +Z / -Z   (up / down)",
        "  Right / Left  +X / -X   (forward / back)",
        "  w / s         +Y / -Y   (left / right)",
        "",
        "  + / -         grow / shrink step",
        "  p             cycle pattern",
        "  SPACE         pause / resume streaming",
        "  q / ESC       quit",
        "",
        f"  step    : {step * 100:.1f} cm",
        f"  pattern : {pattern}",
        f"  active  : {'ON' if active else 'PAUSED'}",
        "",
        f"  centre  : {_fmt_center(center())}",
        "",
        f"  {status}",
    ]
    for i, line in enumerate(lines):
        if i + 2 < height - 1:
            stdscr.addstr(i + 2, 2, line[: max(0, width - 3)])
    stdscr.refresh()


def _jog_loop(stdscr: Any) -> None:
    """curses arrow-key loop: continuously nudge the centre and cycle patterns."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # ms; redraws even with no key so state stays fresh

    step = _DEFAULT_STEP
    active = True
    pattern_i = _PATTERN_CYCLE.index("line") if "line" in _PATTERN_CYCLE else 0
    pattern = _PATTERN_CYCLE[pattern_i]
    status = "Ready. Arrow keys to jog, p to cycle pattern."
    _draw_ui(stdscr, step, pattern, active, status)

    while True:
        key = stdscr.getch()
        if key == -1:  # timeout, no key -- refresh
            _draw_ui(stdscr, step, pattern, active, status)
            continue

        if key in (27, 3, ord("q")):  # ESC, Ctrl-C, q
            break

        key_char = chr(key).lower() if 0 <= key < 256 else None

        if key == curses.KEY_UP:
            status = nudge(dz=step)
        elif key == curses.KEY_DOWN:
            status = nudge(dz=-step)
        elif key == curses.KEY_RIGHT:
            status = nudge(dx=step)
        elif key == curses.KEY_LEFT:
            status = nudge(dx=-step)
        elif key_char == "w":
            status = nudge(dy=step)
        elif key_char == "s":
            status = nudge(dy=-step)
        elif key_char in ("+", "="):
            step = min(_MAX_STEP, round(step + 0.005, 4))
            status = f"step = {step * 100:.1f} cm"
        elif key_char in ("-", "_"):
            step = max(_MIN_STEP, round(step - 0.005, 4))
            status = f"step = {step * 100:.1f} cm"
        elif key_char == "p":
            pattern_i = (pattern_i + 1) % len(_PATTERN_CYCLE)
            pattern = _PATTERN_CYCLE[pattern_i]
            status = set_pattern(pattern)
        elif key_char == " ":
            active = not active
            status = set_active(active)

        _draw_ui(stdscr, step, pattern, active, status)


def jog() -> None:
    """Start the interactive arrow-key jog loop (blocks until you quit)."""
    try:
        curses.wrapper(_jog_loop)
    except Exception as exc:  # surface curses/RPC errors on exit
        traceback.print_exc()
        print(f"jog loop exited with error: {exc}")


if __name__ == "__main__":
    jog()
