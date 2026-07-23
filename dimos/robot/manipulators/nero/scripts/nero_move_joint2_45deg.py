#!/usr/bin/env python3

"""Connect to an AgileX NERO arm and move joint 2 by +45 degrees.

How to run:
1. Bring up the CAN interface, for example:
   sudo ip link set can0 up type can bitrate 1000000

2. Make sure pyAgxArm is installed in the Python environment you use to run this
   script.

3. Run from the repository root:
   python3 dimos/robot/manipulators/nero/scripts/nero_move_joint2_45deg.py

Notes:
- Change CAN_CHANNEL if your arm is on can1 or another CAN interface.
- Change FIRMWARE to match your NERO firmware.
- This sends a full 7-joint move_j target because the AgileX SDK expects all
  joint positions, even when only changing one joint.
"""

from __future__ import annotations

import math
import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


CAN_CHANNEL = "can0"
FIRMWARE = NeroFW.V120
JOINT_INDEX = 1  # Joint 2 in zero-based Python indexing.
DELTA_DEGREES = 45.0


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

    try:
        if robot.has_comm_error():
            raise RuntimeError(f"CAN communication error: {robot.get_comm_error()}")

        while not robot.enable():
            print("Waiting for NERO enable...")
            time.sleep(0.1)

        joint_msg = robot.get_joint_angles()
        if joint_msg is None:
            raise RuntimeError("Could not read current joint angles")

        joints = [float(joint) for joint in joint_msg.msg]
        print("Current joints:", joints)

        joints[JOINT_INDEX] += math.radians(DELTA_DEGREES)
        print("Target joints:", joints)

        robot.move_j(joints)
        time.sleep(3.0)

    finally:
        robot.disable()
        robot.disconnect()


if __name__ == "__main__":
    main()
