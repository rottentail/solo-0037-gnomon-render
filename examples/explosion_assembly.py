"""Exploded assembly walkthrough: plan, queryable before/after positions and orthographic proofs.

Writes real service outputs without needing the HTTP server:
  proof-sheet.png      before/after orthographic views with connector guides
  plan.json            per-node before/after origins, shifts, groups and connector lines
  exploded-scene.json  independent scene payload, ready for the normal render pipeline
"""
import json
import tempfile
from pathlib import Path
from gnomon.service import RendererService
from gnomon.samples import hub_mesh, cap_mesh, bolt_mesh, assembly_scene


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-explosion-") as root:
        service = RendererService(root)
        try:
            hub = service.add_mesh(hub_mesh())
            cap = service.add_mesh(cap_mesh())
            bolt = service.add_mesh(bolt_mesh())
            scene = service.add_scene(assembly_scene(hub["id"], cap["id"], bolt["id"]))
            request = {"scene": scene["id"],
                       "groups": ["carrier", "bolt_a", "bolt_b"],
                       "directions": {"carrier": [1, 0, 0]},
                       "gap_factor": 0.4, "spacing": 0.0,
                       "size": 160, "samples": 1, "angles": [0, 60, 120],
                       "guides": True}
            explosion = service.explode(request)
            for name in explosion["details"]["artifacts"]:
                (output / name).write_bytes(service.artifacts.read(explosion, name))
            summary = {"explosion": explosion["id"], "state": explosion["state"],
                       "groups": explosion["details"]["group_count"],
                       "moved_nodes": explosion["details"]["moved_nodes"],
                       "queries": [
                           {"node": entry["key"], "group": entry["group"],
                            "origin_before": entry["origin_before"],
                            "origin_after": entry["origin_after"],
                            "world_shift": entry["world_shift"]}
                           for entry in explosion["details"]["plan"]["nodes"]]}
            (output / "query-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            print(json.dumps({"explosion": explosion["id"], "output": str(output),
                              "artifacts": sorted(explosion["details"]["artifacts"])},
                             ensure_ascii=False, indent=2))
        finally:
            service.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run-output/explosion"))
    produce(parser.parse_args().output)
