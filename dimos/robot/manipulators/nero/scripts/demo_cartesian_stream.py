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

"""World-frame cartesian pattern streamer for the NERO cartesian_ik coordinators.

Generates a repeating end-effector path (line or circle) centred at a WORLD-frame
point and streams it to a running NERO cartesian_ik coordinator. Targets are
transformed from world into the arm's base frame (the control-side Pinocchio
model has base_link at the origin) and published on /coordinator_cartesian_command
with frame_id == the arm's task name, so the coordinator routes them to the right
arm. The cartesian_ik task solves IK each tick from the live joints and drives
the arm over CPV.

WARNING: this control path performs NO collision or joint-limit avoidance. Keep
the centre + amplitude/radius inside a known-safe workspace. Validate on the mock
coordinator (Viser) before running on real hardware.

Prerequisite - a NERO cartesian coordinator running in another terminal:

    dimos run coordinator-nero-cartesian-mock      # simulate (Viser only)
    dimos run coordinator-nero-cartesian-left      # real left arm (can0)

Examples:

    # Up/down 5 cm line along world Z, 3 s period, centred at current EE:
    python -m dimos.robot.manipulators.nero.scripts.demo_cartesian_stream \
        --arm left_arm --pattern line --axis z --amplitude 0.05 --period 3

    # 4 cm circle in the world XY plane, centred at an explicit world point:
    python -m dimos.robot.manipulators.nero.scripts.demo_cartesian_stream \
        --arm left_arm --pattern circle --plane xy --radius 0.04 --period 4 \
        --center 0.40 0.10 0.35

Open the coordinator's Viser URL (printed in its log, default http://127.0.0.1:8095)
to watch the arm track the pattern.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.manipulators._modeling import base_pose
from dimos.robot.manipulators.nero.blueprints.cartesian import (
    CARTESIAN_IK_LEFT_TASK,
    CARTESIAN_IK_RIGHT_TASK,
)
from dimos.robot.manipulators.nero.config import (
    NERO_BASE_X,
    NERO_BASE_Z,
    NERO_LEFT_BASE_RPY,
    NERO_LEFT_BASE_Y,
    NERO_RIGHT_BASE_RPY,
    NERO_RIGHT_BASE_Y,
)
from dimos.utils.transform_utils import matrix_to_pose, pose_to_matrix

COMMAND_TOPIC = "/coordinator_cartesian_command"
_AXES = {"x": 0, "y": 1, "z": 2}
_PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _arm_task_and_base(arm: str) -> tuple[str, np.ndarray]:
    """Return (task_name, T_world_base 4x4) for the requested arm."""
    if arm == "left_arm":
        task = CARTESIAN_IK_LEFT_TASK
        bp = base_pose(
            x=NERO_BASE_X, y=NERO_LEFT_BASE_Y, z=NERO_BASE_Z,
            roll=NERO_LEFT_BASE_RPY[0], pitch=NERO_LEFT_BASE_RPY[1], yaw=NERO_LEFT_BASE_RPY[2],
        )
    elif arm == "right_arm":
        task = CARTESIAN_IK_RIGHT_TASK
        bp = base_pose(
            x=NERO_BASE_X, y=NERO_RIGHT_BASE_Y, z=NERO_BASE_Z,
            roll=NERO_RIGHT_BASE_RPY[0], pitch=NERO_RIGHT_BASE_RPY[1], yaw=NERO_RIGHT_BASE_RPY[2],
        )
    else:
        raise ValueError(f"arm must be 'left_arm' or 'right_arm', got {arm!r}")
    return task, pose_to_matrix(bp)


def _seed_center(arm: str) -> tuple[np.ndarray, Quaternion] | None:
    """Read the current world-frame EE pose (position, orientation) over RPC."""
    try:
        from dimos.core.rpc_client import RPCClient
        from dimos.manipulation.manipulation_module import ManipulationModule

        client = RPCClient(None, ManipulationModule)
        try:
            pose = client.get_ee_pose(arm)
        finally:
            client.stop_rpc_client()
        if pose is None:
            return None
        center = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        orient = Quaternion(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        return center, orient
    except Exception as exc:
        print(f"(could not read current EE pose over RPC: {exc})")
        return None


def _offset(pattern: str, phase: float, args: argparse.Namespace) -> np.ndarray:
    """World-frame position offset for the current phase (radians)."""
    off = np.zeros(3, dtype=float)
    if pattern == "line":
        off[_AXES[args.axis]] = args.amplitude * math.sin(phase)
    else:  # circle
        i, j = _PLANES[args.plane]
        off[i] = args.radius * math.cos(phase)
        off[j] = args.radius * math.sin(phase)
    return off


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="left_arm", choices=["left_arm", "right_arm"])
    parser.add_argument("--pattern", default="line", choices=["line", "circle"])
    parser.add_argument("--axis", default="z", choices=list(_AXES), help="line axis (world)")
    parser.add_argument("--plane", default="xy", choices=list(_PLANES), help="circle plane (world)")
    parser.add_argument("--amplitude", type=float, default=0.05, help="line half-stroke (m)")
    parser.add_argument("--radius", type=float, default=0.04, help="circle radius (m)")
    parser.add_argument("--period", type=float, default=3.0, help="seconds per cycle")
    parser.add_argument("--rate", type=float, default=50.0, help="publish rate (Hz)")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds (0 = forever)")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="world-frame centre; default = current EE position")
    args = parser.parse_args()

    task_name, t_world_base = _arm_task_and_base(args.arm)
    t_base_world = np.linalg.inv(t_world_base)

    # Determine centre + held orientation.
    seed = _seed_center(args.arm)
    if args.center is not None:
        center = np.array(args.center, dtype=float)
        orientation = seed[1] if seed else Quaternion(0.0, 0.0, 0.0, 1.0)
        if seed is None:
            print("WARNING: no live EE orientation; using identity. Prefer seeding from the arm.")
    elif seed is not None:
        center, orientation = seed
    else:
        print("ERROR: no --center given and could not read current EE pose. "
              "Is a NERO cartesian coordinator running?")
        return 1

    print(f"arm={args.arm} task={task_name} pattern={args.pattern}")
    print(f"center(world)={np.round(center, 4).tolist()}  period={args.period}s rate={args.rate}Hz")
    print(f"Publishing to {COMMAND_TOPIC} (frame_id={task_name}). Ctrl-C to stop.")
    if args.pattern == "line":
        print(f"line: {args.amplitude * 100:.1f} cm along world {args.axis}")
    else:
        print(f"circle: r={args.radius * 100:.1f} cm in world {args.plane} plane")

    transport: LCMTransport[PoseStamped] = LCMTransport(COMMAND_TOPIC, PoseStamped)
    dt = 1.0 / args.rate
    t0 = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - t0
            if args.duration > 0 and t >= args.duration:
                break
            phase = 2.0 * math.pi * (t / args.period)
            world_pos = center + _offset(args.pattern, phase, args)

            world_pose = Pose(
                position=Vector3(x=world_pos[0], y=world_pos[1], z=world_pos[2]),
                orientation=orientation,
            )
            t_base_target = t_base_world @ pose_to_matrix(world_pose)
            base_target = matrix_to_pose(t_base_target)

            transport.publish(
                PoseStamped(
                    frame_id=task_name,
                    position=base_target.position,
                    orientation=base_target.orientation,
                )
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped streaming. The task will time out (0.5 s) and hold position.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
