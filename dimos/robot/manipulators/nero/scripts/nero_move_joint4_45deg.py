#!/usr/bin/env python3

"""Connect to an AgileX NERO arm and move joint 4 by +45 degrees.

How to run:
1. Bring up the CAN interface, for example:
   sudo ip link set can0 up type can bitrate 1000000

2. Make sure pyAgxArm is installed in the Python environment you use to run this
   script.

3. Run from the repository root:
   python3 dimos/robot/manipulators/nero/scripts/nero_move_joint4_45deg.py

Notes:
- Change CAN_CHANNEL if your arm is on can1 or another CAN interface.
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


CAN_CHANNEL = "can0"
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


def wait_for_motion(robot, timeout_s: float = 10.0) -> None:
    start_t = time.monotonic()
    while time.monotonic() - start_t < timeout_s:
        status = robot.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            print("Reached target position")
            return
        time.sleep(0.1)
    print(f"Wait for motion timeout ({timeout_s:.1f}s)")


def hold_enabled_until_interrupt(message: str) -> None:
    print(message)
    while True:
        time.sleep(1.0)


def main() -> None:
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=FIRMWARE,
        interface="socketcan",
        channel=CAN_CHANNEL,
        bitrate=1_000_000,
    )

    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    enabled = False
    try:
        if robot.has_comm_error():
            raise RuntimeError(f"CAN communication error: {robot.get_comm_error()}")

        while not robot.enable():
            print("Waiting for NERO enable...")
            time.sleep(0.1)
        enabled = True

        joint_msg = robot.get_joint_angles()
        if joint_msg is None:
            raise RuntimeError("Could not read current joint angles")

        joints = [float(joint) for joint in joint_msg.msg]
        print("Current joints:", joints)

        joints[JOINT_INDEX] += math.radians(DELTA_DEGREES)
        lower, upper = JOINT_LIMITS_RAD[JOINT_INDEX]
        if not lower <= joints[JOINT_INDEX] <= upper:
            raise RuntimeError(
                f"Joint 4 target {joints[JOINT_INDEX]:.3f} rad is outside "
                f"NERO limits [{lower:.3f}, {upper:.3f}] rad"
            )
        print("Target joints:", joints)

        robot.move_j(joints)
        wait_for_motion(robot)

        hold_enabled_until_interrupt(
            "Arm is enabled, connected, and holding position. "
            "Press Ctrl+C to command zero."
        )

    except KeyboardInterrupt:
        if not enabled:
            print("Interrupted before enable completed. No zero command was sent.")
            return
        print("Returning to zero while staying enabled.")
        robot.move_j(ZERO_JOINTS)
        wait_for_motion(robot)
        try:
            hold_enabled_until_interrupt(
                "Zero target sent. Arm is still enabled and connected. "
                "Press Ctrl+C again only when safe to exit this process."
            )
        except KeyboardInterrupt:
            print("Exiting without disable() or disconnect().")


if __name__ == "__main__":
    main()
