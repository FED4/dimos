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

"""Identify WHICH body pairs collide in the NERO planning world.

The NERO planner registers BOTH arms in one Drake world, and collision checking
is global (query_object.HasCollisions()). So an IK/plan can be rejected because
of a self-collision, an inter-arm collision, or a robot-vs-table collision. This
script rebuilds the same two-arm world, sets each arm to a chosen configuration,
and prints every penetrating geometry pair by name + penetration depth.

Prerequisite (optional): a running NERO planner blueprint, so we can read the
arms' live joint states over RPC:

    dimos run nero-mock-planner-coordinator     # terminal 1

Then:

    # Inspect at the arms' current LIVE joint states (read via RPC):
    python -m dimos.robot.manipulators.nero.scripts.demo_collision_inspect

    # Inspect at the home config (all zeros) for both arms (no RPC needed):
    python -m dimos.robot.manipulators.nero.scripts.demo_collision_inspect --home
"""

from __future__ import annotations

import argparse

from dimos.manipulation.planning.factory import create_world
from dimos.manipulation.planning.spec.enums import ObstacleType
from dimos.manipulation.planning.spec.models import Obstacle
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.nero.blueprints.planner import left_model, right_model
from dimos.robot.manipulators.nero.config import (
    NERO_DOF,
    nero_default_table_obstacle,
)


def _table_obstacle() -> Obstacle:
    cfg = nero_default_table_obstacle()
    x, y, z = cfg["position"]
    return Obstacle(
        name=cfg["name"],
        obstacle_type=ObstacleType.BOX,
        pose=PoseStamped(
            position=Vector3(x=x, y=y, z=z),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
        dimensions=tuple(cfg["dimensions"]),
    )


def _live_joints() -> dict[str, list[float]] | None:
    """Read both arms' current joint states from a running blueprint via RPC."""
    try:
        from dimos.core.rpc_client import RPCClient
        from dimos.manipulation.manipulation_module import ManipulationModule

        client = RPCClient(None, ManipulationModule)
        try:
            left = client.get_current_joints("left_arm")
            right = client.get_current_joints("right_arm")
        finally:
            client.stop_rpc_client()
        if left is None or right is None:
            return None
        return {"left_arm": list(left), "right_arm": list(right)}
    except Exception as exc:  # noqa: BLE001 - diagnostic best-effort
        print(f"(could not read live joints over RPC: {exc})")
        return None


def _body_label(plant: object, query_inspector: object, geometry_id: object) -> str:
    """Map a Drake GeometryId to 'model_instance / body' for readability."""
    try:
        frame_id = query_inspector.GetFrameId(geometry_id)  # type: ignore[attr-defined]
        body = plant.GetBodyFromFrameId(frame_id)  # type: ignore[attr-defined]
        model = plant.GetModelInstanceName(body.model_instance())  # type: ignore[attr-defined]
        return f"{model}/{body.name()}"
    except Exception:  # noqa: BLE001
        return str(query_inspector.GetName(geometry_id))  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", action="store_true",
                        help="Use home config (all zeros) for both arms instead of live state")
    args = parser.parse_args()

    if args.home:
        joints = {"left_arm": [0.0] * NERO_DOF, "right_arm": [0.0] * NERO_DOF}
        print("Using HOME config (all zeros) for both arms.")
    else:
        live = _live_joints()
        if live is None:
            print("No live joints available; falling back to HOME (all zeros). "
                  "Start a blueprint or pass --home.")
            joints = {"left_arm": [0.0] * NERO_DOF, "right_arm": [0.0] * NERO_DOF}
        else:
            joints = live
            print(f"left_arm  joints: {[round(j, 4) for j in joints['left_arm']]}")
            print(f"right_arm joints: {[round(j, 4) for j in joints['right_arm']]}")

    print("\nBuilding two-arm Drake world (this loads/*converts* meshes, may take a bit)...")
    world = create_world("drake")
    left_id = world.add_robot(left_model)
    right_id = world.add_robot(right_model)
    world.add_obstacle(_table_obstacle())
    world.finalize()
    ids = {"left_arm": left_id, "right_arm": right_id}

    plant = world._plant  # noqa: SLF001 - diagnostic introspection
    scene_graph = world._scene_graph  # noqa: SLF001
    diagram = world._diagram  # noqa: SLF001

    with world.scratch_context() as ctx:
        for name, rid in ids.items():
            world.set_joint_state(
                ctx, rid, JointState(name=list(left_model.joint_names), position=joints[name])
            )

        scene_graph_ctx = diagram.GetSubsystemContext(scene_graph, ctx)
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_ctx)
        inspector = query_object.inspector()

        has = query_object.HasCollisions()
        print(f"\nHasCollisions() = {has}")

        pairs = query_object.ComputePointPairPenetration()
        if not pairs:
            if has:
                print("HasCollisions() is True but point-pair query returned nothing "
                      "(likely non-convex mesh contact). Try get_min_distance / signed-distance.")
                _report_signed_distance(query_object, plant, inspector)
            else:
                print("No penetrating pairs. Configuration is collision-free.")
            return 0

        print(f"\n{len(pairs)} penetrating pair(s) (deepest first):")
        rows = []
        for p in pairs:
            a = _body_label(plant, inspector, p.id_A)
            b = _body_label(plant, inspector, p.id_B)
            rows.append((float(p.depth), a, b))
        for depth, a, b in sorted(rows, key=lambda r: -r[0]):
            kind = _classify(a, b)
            print(f"  depth={depth * 1000:7.2f} mm  {a:28s} <-> {b:28s}  [{kind}]")

    return 0


def _report_signed_distance(query_object: object, plant: object, inspector: object) -> None:
    """Fallback: list closest/penetrating pairs via signed-distance query."""
    try:
        sdps = query_object.ComputeSignedDistancePairwiseClosestPoints(0.0)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        print(f"(signed-distance query failed: {exc})")
        return
    rows = []
    for pair in sdps:
        if float(pair.distance) < 0.0:
            a = _body_label(plant, inspector, pair.id_A)
            b = _body_label(plant, inspector, pair.id_B)
            rows.append((float(pair.distance), a, b))
    if not rows:
        print("(no negative signed-distance pairs found)")
        return
    print(f"{len(rows)} penetrating pair(s) via signed distance (deepest first):")
    for dist, a, b in sorted(rows, key=lambda r: r[0]):
        print(f"  depth={-dist * 1000:7.2f} mm  {a:28s} <-> {b:28s}  [{_classify(a, b)}]")


def _classify(a: str, b: str) -> str:
    ma = a.split("/", 1)[0]
    mb = b.split("/", 1)[0]
    if "table" in a.lower() or "table" in b.lower():
        return "robot-vs-table"
    if ma == mb:
        return "self-collision"
    return "inter-arm"


if __name__ == "__main__":
    raise SystemExit(main())
