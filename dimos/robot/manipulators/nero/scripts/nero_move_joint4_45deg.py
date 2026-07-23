#!/usr/bin/env python3

"""Connect to AgileX NERO arms on can0/can1 and move joint 4 by +45 degrees.

How to run:
1. Bring up the CAN interface, for example:
   sudo ip link set can0 up type can bitrate 1000000
   sudo ip link set can1 up type can bitrate 1000000

2. Make sure pyAgxArm is installed in the Python environment you use to run this
   script.

3. Run from the repository root:
   python3 dimos/robot/manipulators/nero/scripts/nero_move_joint4_45deg.py

Notes:
- Change CAN_CHANNELS if your arms use different CAN interfaces.
- Change FIRMWARE to match your NERO firmware.
- This sends a full 7-joint move_j target because the AgileX SDK expects all
  joint positions, even when only changing one joint.
- Do not call disable() while the arm is raised. AgileX documents that disabled
  NERO joints can drop immediately because servo holding torque is removed.
- This script intentionally keeps the SDK connection open after the move.
  Press Ctrl+C once to command the arm back to zero. After zero is reached, the
  script keeps holding enabled/connected; press Ctrl+C again only when safe.
"""

from __future__ import annotations

import math
import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


CAN_CHANNELS = ["can0", "can1"]
FIRMWARE = NeroFW.V120
JOINT_INDEX = 3  # Joint 4 in zero-based Python indexing.
DELTA_DEGREES = 45.0
ZERO_JOINTS = [0.0] * 7

JOINT_LIMITS_RAD = [
    (-2.705261, 2.705261),
    (-1.745330, 1.745330),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
]


def wait_for_motion(channel: str, robot, timeout_s: float = 10.0) -> None:
    start_t = time.monotonic()
    while time.monotonic() - start_t < timeout_s:
        status = robot.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            print(f"{channel}: reached target position")
            return
        time.sleep(0.1)
    print(f"{channel}: wait for motion timeout ({timeout_s:.1f}s)")


def hold_enabled_until_interrupt(message: str) -> None:
    print(message)
    while True:
        time.sleep(1.0)


def create_robot(channel: str):
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=FIRMWARE,
        interface="socketcan",
        channel=channel,
        bitrate=1_000_000,
    )
    return AgxArmFactory.create_arm(cfg)


def connect_and_enable(channel: str):
    robot = create_robot(channel)
    robot.connect()

    if robot.has_comm_error():
        raise RuntimeError(f"{channel}: CAN communication error: {robot.get_comm_error()}")

    while not robot.enable():
        print(f"{channel}: waiting for NERO enable...")
        time.sleep(0.1)
    return robot


def move_joint4_delta(channel: str, robot) -> None:
    joint_msg = robot.get_joint_angles()
    if joint_msg is None:
        raise RuntimeError(f"{channel}: could not read current joint angles")

    joints = [float(joint) for joint in joint_msg.msg]
    print(f"{channel}: current joints:", joints)

    joints[JOINT_INDEX] += math.radians(DELTA_DEGREES)
    lower, upper = JOINT_LIMITS_RAD[JOINT_INDEX]
    if not lower <= joints[JOINT_INDEX] <= upper:
        raise RuntimeError(
            f"{channel}: joint 4 target {joints[JOINT_INDEX]:.3f} rad is outside "
            f"NERO limits [{lower:.3f}, {upper:.3f}] rad"
        )
    print(f"{channel}: target joints:", joints)

    robot.move_j(joints)


def main() -> None:
    robots: dict[str, object] = {}

    try:
        for channel in CAN_CHANNELS:
            robots[channel] = connect_and_enable(channel)

        for channel, robot in robots.items():
            move_joint4_delta(channel, robot)

        for channel, robot in robots.items():
            wait_for_motion(channel, robot)

        hold_enabled_until_interrupt(
            "Arms are enabled, connected, and holding position. "
            "Press Ctrl+C to command zero."
        )

    except KeyboardInterrupt:
        if not robots:
            print("Interrupted before enable completed. No zero command was sent.")
            return
        print("Returning connected arms to zero while staying enabled.")
        for channel, robot in robots.items():
            print(f"{channel}: sending zero target")
            robot.move_j(ZERO_JOINTS)
        for channel, robot in robots.items():
            wait_for_motion(channel, robot)
        try:
            hold_enabled_until_interrupt(
                "Zero target sent. Arms are still enabled and connected. "
                "Press Ctrl+C again only when safe to exit this process."
            )
        except KeyboardInterrupt:
            print("Exiting without disable() or disconnect().")


if __name__ == "__main__":
    main()
