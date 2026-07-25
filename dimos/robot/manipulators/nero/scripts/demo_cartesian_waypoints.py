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

"""Cartesian waypoint driver for the AgileX NERO arm (plan-then-execute IK).

This is the "in-place" motion method: for each Cartesian waypoint it runs
IK + collision-free RRT planning (ManipulationModule.plan_to_pose), optionally
previews the trajectory in Viser, then executes it through the ControlCoordinator
trajectory task (traj_<arm>). Each waypoint is a separate planned motion; the arm
comes to rest at every waypoint. This is NOT continuous servoing.

Prerequisite — start a NERO planner+coordinator blueprint in another terminal:

    # Mock (no hardware / no CAN, visualise in Viser):
    dimos run nero-mock-planner-coordinator

    # Real hardware (drives the physical arm over CAN):
    dimos run nero-planner-coordinator

Then run this script (from the dimos repo root, in the same env):

    # 1. Probe only: print current EE pose + joints, plan nothing, move nothing.
    python -m dimos.robot.manipulators.nero.scripts.demo_cartesian_waypoints --probe

    # 2. Dry run: plan every waypoint and preview in Viser, but DO NOT move.
    python -m dimos.robot.manipulators.nero.scripts.demo_cartesian_waypoints

    # 3. Execute: plan + move the arm through the waypoints.
    python -m dimos.robot.manipulators.nero.scripts.demo_cartesian_waypoints --execute

Workflow: run --probe first to read the current world-frame EE pose, copy those
numbers into WAYPOINTS below (they are ABSOLUTE world-frame coordinates), then do
a --preview/dry run, and only then --execute.
"""

from __future__ import annotations

import argparse
import math
import time

from dimos.core.rpc_client import RPCClient
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3

# ---------------------------------------------------------------------------
# Waypoints — ABSOLUTE world-frame end-effector targets.
#
# Each entry is (x, y, z, roll, pitch, yaw):
#   x, y, z            metres, world frame
#   roll, pitch, yaw   radians; set any of them to None to KEEP the arm's
#                      current EE orientation (recommended until you know a
#                      good orientation for your workspace).
#
# Leave this list EMPTY to auto-generate a safe small square around the CURRENT
# EE pose (printed at startup). Once you know good coordinates from --probe,
# replace it with your own absolute poses, e.g.:
#
#   WAYPOINTS = [
#       (0.35, 0.10, 0.45, None, None, None),
#       (0.40, 0.10, 0.45, None, None, None),
#       (0.40, 0.10, 0.40, None, None, None),
#   ]
# ---------------------------------------------------------------------------
WAYPOINTS: list[tuple[float, float, float, float | None, float | None, float | None]] = []

# Size of the auto-generated square (metres) used only when WAYPOINTS is empty.
_AUTO_STEP = 0.25

# Poll settings while waiting for a motion to finish.
_EXECUTE_TIMEOUT_S = 30.0
_POLL_INTERVAL_S = 0.1
_TERMINAL_STATES = {"IDLE", "COMPLETED"}


def _build_orientation(
    base: Pose,
    roll: float | None,
    pitch: float | None,
    yaw: float | None,
) -> Quaternion:
    """Use explicit rpy if any component is given, else keep base orientation."""
    if roll is None and pitch is None and yaw is None:
        return base.orientation
    return Quaternion.from_euler(
        Vector3(x=roll or 0.0, y=pitch or 0.0, z=yaw or 0.0)
    )


def _auto_waypoints(
    base: Pose,
) -> list[tuple[float, float, float, float | None, float | None, float | None]]:
    """A small square in the world XY plane around the current EE pose."""
    x, y, z = base.position.x, base.position.y, base.position.z
    s = _AUTO_STEP
    return [
        (x + s, y, z, None, None, None),
        #(x + s, y + s, z, None, None, None),
        #(x, y + s, z, None, None, None),
        #(x, y, z, None, None, None),  # back to start
    ]


def _fmt_pose(pose: Pose) -> str:
    p = pose.position
    q = pose.orientation
    return (
        f"pos=({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f}) "
        f"quat=({q.x:+.3f}, {q.y:+.3f}, {q.z:+.3f}, {q.w:+.3f})"
    )


def _ensure_ready(client: RPCClient) -> None:
    """Clear any latched FAULT/PLANNING state left by a previous run.

    ManipulationModule latches into FAULT after a failed plan/execute and then
    refuses new IK/plan calls ("Cannot solve IK while state is FAULT") until
    reset() is called. Calling reset() here makes the script robust to prior
    failures. reset() is refused while EXECUTING, so cancel first in that case.
    """
    state = client.get_state()
    if state == "EXECUTING":
        print(f"State={state}; cancelling before reset...")
        client.cancel()
    if state not in ("IDLE", "COMPLETED"):
        client.reset()
        print(f"State was {state} -> reset -> {client.get_state()}")


def _describe_ik(result: object) -> str:
    """Format an IKResult (returned over RPC) defensively."""
    status = getattr(getattr(result, "status", None), "name", str(getattr(result, "status", "?")))
    pos = getattr(result, "position_error", None)
    ori = getattr(result, "orientation_error", None)
    msg = getattr(result, "message", "") or ""
    pos_s = f"{pos:.4f}" if isinstance(pos, (int, float)) else str(pos)
    ori_s = f"{ori:.4f}" if isinstance(ori, (int, float)) else str(ori)
    return f"status={status} pos_err={pos_s} ori_err={ori_s} msg={msg}"


def diagnose(client: RPCClient, arm: str, base: Pose) -> int:
    """Separate reachability (IK w/o collision) from collision rejection.

    This answers: are the Cartesian poses simply unreachable, or is the
    collision check throwing out otherwise-valid IK solutions?
    """
    joints = client.get_current_joints(arm) or []
    start_free = client.is_collision_free(list(joints), arm)
    print(f"\n[diagnose] current config collision-free? {start_free}")
    if not start_free:
        print("  -> The START pose itself is flagged in collision. Planning (RRT) will")
        print("     always fail at the start state until the collision model is fixed.")
        print("     This is a model issue (missing self-collision exclusions, coarse")
        print("     collision meshes, or the startup table obstacle), NOT reachability.")

    waypoints = WAYPOINTS or _auto_waypoints(base)
    print(f"\n[diagnose] testing {len(waypoints)} waypoint(s): IK without vs with collision\n")
    for i, (x, y, z, roll, pitch, yaw) in enumerate(waypoints, start=1):
        target = Pose(
            position=Vector3(x=x, y=y, z=z),
            orientation=_build_orientation(base, roll, pitch, yaw),
        )
        no_col = client.solve_ik(target, arm, False, None)   # reachability only
        with_col = client.solve_ik(target, arm, True, None)  # + collision gate
        print(f"Waypoint {i}: {_fmt_pose(target)}")
        print(f"  no-collision : {_describe_ik(no_col)}")
        print(f"  collision    : {_describe_ik(with_col)}")
    print("\nInterpretation:")
    print("  * no-collision SUCCESS + collision COLLISION  -> reachable, collision model")
    print("    is the blocker (relax/fix collision).")
    print("  * no-collision NO_SOLUTION/JOINT_LIMITS        -> genuinely unreachable or")
    print("    orientation infeasible (change target or relax orientation).")
    return 0


def _is_stale_plan_error(err: str) -> bool:
    e = err.lower()
    return any(s in e for s in ("no longer matches", "stored plan start", "stale", "freshness"))


def _execute_planned(client: RPCClient, arm: str, target: Pose) -> bool:
    """Execute the stored plan; on a stale-plan rejection, re-plan and retry once.

    A stale-plan rejection means the arm's reported joints differ from the
    trajectory's start (tolerance 1e-6 rad). We re-plan from the current state
    and execute immediately (no preview gap).
    """
    for attempt in (1, 2):
        if client.execute(arm):
            return _wait_until_idle(client, _EXECUTE_TIMEOUT_S)
        err = client.get_error() or ""
        cur = client.get_current_joints(arm)
        print(f"  ! execute rejected (attempt {attempt}): {err}")
        print(f"    current joints: {[round(j, 5) for j in (cur or [])]}")
        client.reset()
        if attempt == 2 or not _is_stale_plan_error(err):
            return False
        print("  re-planning from current state and retrying immediately...")
        if not client.plan_to_pose(target, arm):
            print(f"  ! re-plan failed: {client.get_error()}")
            client.reset()
            return False
    return False


def _wait_until_idle(client: RPCClient, timeout: float) -> bool:
    """Poll get_state() until the module leaves EXECUTING (or times out)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get_state()
        if state in _TERMINAL_STATES:
            return True
        time.sleep(_POLL_INTERVAL_S)
    print(f"  ! timed out after {timeout:.0f}s waiting for motion to finish")
    return False


def run(arm: str, *, do_preview: bool, do_execute: bool, probe: bool, do_diagnose: bool) -> int:
    client = RPCClient(None, ManipulationModule)
    try:
        robots = client.list_robots()
        print(f"Configured robots: {robots}")
        if arm not in robots:
            print(f"ERROR: arm '{arm}' not in configured robots {robots}")
            return 1

        _ensure_ready(client)

        joints = client.get_current_joints(arm)
        base = client.get_ee_pose(arm)
        if base is None:
            print(f"ERROR: could not read EE pose for '{arm}'. Is the blueprint running?")
            return 1
        print(f"[{arm}] current joints: {[round(j, 4) for j in (joints or [])]}")
        print(f"[{arm}] current EE   : {_fmt_pose(base)}")

        url = client.get_visualization_url()
        if url:
            print(f"Viser visualization: {url}")

        if probe:
            print("\n--probe: nothing planned, nothing moved. "
                  "Copy the EE pos above into WAYPOINTS.")
            return 0

        if do_diagnose:
            return diagnose(client, arm, base)

        waypoints = WAYPOINTS or _auto_waypoints(base)
        if not WAYPOINTS:
            print(f"\nWAYPOINTS empty -> auto square (step {_AUTO_STEP} m) around current pose.")

        mode = "EXECUTE (arm will move)" if do_execute else "DRY RUN (plan only, no motion)"
        print(f"\nMode: {mode}. {len(waypoints)} waypoint(s).\n")

        for i, (x, y, z, roll, pitch, yaw) in enumerate(waypoints, start=1):
            target = Pose(
                position=Vector3(x=x, y=y, z=z),
                orientation=_build_orientation(base, roll, pitch, yaw),
            )
            print(f"Waypoint {i}/{len(waypoints)} -> {_fmt_pose(target)}")

            if not client.plan_to_pose(target, arm):
                err = client.get_error()
                print(f"  ! planning FAILED{f': {err}' if err else ''} -> aborting")
                client.reset()  # clear FAULT so the next run / web UI starts clean
                return 1
            print("  planned OK")

            # Preview only in dry-run mode. During execution the preview adds a
            # multi-second window between plan and execute; the coordinator
            # rejects a stored plan whose start no longer matches current state
            # (tolerance 1e-6), so we execute immediately after planning instead.
            if do_preview and not do_execute:
                client.preview_plan(None, None, arm)
                print("  previewed in Viser")

            if not do_execute:
                continue

            if not _execute_planned(client, arm, target):
                return 1
            print(f"  executed (state={client.get_state()})")

        print("\nDone.")
        return 0
    finally:
        client.stop_rpc_client()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="left_arm", help="Robot name (left_arm | right_arm)")
    parser.add_argument("--probe", action="store_true",
                        help="Print current EE pose + joints and exit (no planning, no motion)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Test each waypoint's IK with and without collision checking "
                             "(no motion). Isolates reachability from collision rejection.")
    parser.add_argument("--preview", dest="preview", action="store_true", default=True,
                        help="Animate each planned trajectory in Viser (default on)")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="Skip Viser preview")
    parser.add_argument("--execute", action="store_true",
                        help="Actually move the arm. Without this, only plan/preview (dry run).")
    args = parser.parse_args()

    raise SystemExit(
        run(
            args.arm,
            do_preview=args.preview,
            do_execute=args.execute,
            probe=args.probe,
            do_diagnose=args.diagnose,
        )
    )


if __name__ == "__main__":
    main()
