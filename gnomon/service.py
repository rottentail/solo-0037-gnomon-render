import json
import threading
import time
from .catalog import Catalog, canonical
from .artifacts import Artifacts
from .mesh import prepare_mesh
from .texture import decode_texture
from .scene import compile_scene, scene_summary
from .raster import render_samples
from .channels import resolve_samples, preview_channels, packed_channels, unpack_channels
from .inspection import inspect_pixel
from .comparison import compare_frames
from .sheet import build_sheet
from .bundle import build_bundle
from . import stereo as stereo_core
from .errors import RenderError, require
from .schema import MeshInput, TextureInput, SceneInput, RenderInput, CompareInput, SheetInput


def _stereo_summary(metadata,checks,analysis):
    directions = analysis["summaries"]
    forward = directions["left_to_right"]
    backward = directions["right_to_left"]
    return {
        "baseline_meters": metadata["units"]["interocular"]["meters"],
        "baseline_declared": f"{metadata['units']['interocular']['value']:g} {metadata['units']['interocular']['unit']}",
        "convergence_meters": metadata["units"]["convergence"]["meters"],
        "convergence_source": metadata["units"]["convergence"]["source"],
        "rig_parallel_and_translation_only": checks["translation_only"],
        "zero_parallax_at_convergence": checks["zero_parallax_within_quarter_pixel"],
        "vertical_parallax_within_tolerance": bool(
            forward["vertical_parallax_pixels"]["within_tolerance"] and
            backward["vertical_parallax_pixels"]["within_tolerance"]),
        "parallax_depth_dependent": forward["parallax_depth_fit"].get("depth_dependent",False),
        "measured_matches_analytic_parallax": bool(
            analysis["probe_checks"]["left_to_right"]["all_within_tolerance"] and
            analysis["probe_checks"]["right_to_left"]["all_within_tolerance"]),
        "covered_pixels_left": forward["source_surface_pixels"],
        "trusted_pixels_left_to_right": forward["trusted_pixels"],
        "trusted_fraction_of_covered_left": forward["trusted_fraction_of_source_surface"],
        "covered_but_untrusted_left_to_right": forward["source_surface_pixels"] - forward["trusted_pixels"],
        "untrusted_status_counts_left_to_right": {
            name:count for name,count in forward["status_counts"].items()
            if name not in {"matched","no_source_surface"}
        },
        "regions_without_trusted_correspondence": forward["regions_without_trusted_correspondence"],
    }


class RendererService:
    def __init__(self,root):
        self.catalog = Catalog(root)
        self.artifacts = Artifacts(root)
        self.operation_lock = threading.RLock()
        self.recovered_operations = self.catalog.recover_interrupted()

    def close(self):
        self.catalog.close()

    def add_mesh(self,payload):
        parsed = MeshInput.model_validate(payload).model_dump()
        return self.catalog.create("mesh",prepare_mesh(parsed))

    def add_texture(self,payload):
        parsed = TextureInput.model_validate(payload).model_dump()
        return self.catalog.create("texture",decode_texture(parsed))

    def add_scene(self,payload,predecessor=None):
        parsed = SceneInput.model_validate(payload).model_dump()
        with self.operation_lock:
            if predecessor:
                self.catalog.get(predecessor,"scene")
            compiled = compile_scene(parsed,self.catalog.get)
            return self.catalog.create("scene",parsed,details={"predecessor":predecessor,**scene_summary(compiled)})

    def scene_state(self,identity,target):
        require(target in {"active","archived"},"scene_state","unknown scene state")
        with self.operation_lock:
            self.catalog.get(identity,"scene")
            expected = {"active"} if target=="archived" else {"archived"}
            return self.catalog.transition(identity,expected,target)

    def resource_state(self,identity,target):
        require(target in {"active","retired"},"resource_state","unknown resource state")
        with self.operation_lock:
            record = self.catalog.get(identity)
            require(record["kind"] in {"mesh","texture"},"resource_kind","only mesh and texture can retire")
            expected = {"active"} if target=="retired" else {"retired"}
            return self.catalog.transition(identity,expected,target)

    def preflight(self,request):
        parsed = RenderInput.model_validate(request).model_dump()
        scene = self.catalog.get(parsed["scene"],"scene")
        require(scene["state"]=="active","scene_archived","scene is archived",409)
        compiled = compile_scene(scene["payload"],self.catalog.get,allow_retired=True)
        region = parsed["region"] or [0,0,parsed["width"],parsed["height"]]
        estimated = region[2]*region[3]*parsed["samples"]**2*128
        eyes = 2 if parsed["stereo"] is not None else 1
        stereo = None
        if parsed["stereo"] is not None:
            stereo_core.build_rig(scene["payload"]["lens"],parsed["stereo"],
                                  parsed["width"]/parsed["height"])
            stereo = {"eyes":eyes,"per_eye_buffer_bytes":estimated,
                      "combined_buffer_bytes":estimated*eyes,
                      "note":"each eye uses the same per-render sample budget; the pair is two independent renders plus a correspondence pass"}
        return {**scene_summary(compiled),"estimated_buffer_bytes":estimated,
                "estimated_total_buffer_bytes":estimated*eyes,"eyes":eyes,"stereo":stereo,
                "admissible":estimated<=160*1024*1024 and compiled["triangles"]<=60000,
                "region":region,"request":parsed}

    def _snapshot(self,scene):
        compiled = compile_scene(scene["payload"],self.catalog.get,allow_retired=True)
        snapshot = {"scene_id":scene["id"],"scene_digest":scene["digest"],"scene":scene["payload"],
                    "resources":{key:{"kind":value["kind"],"digest":value["digest"],"payload":value["payload"],"state":value["state"]} for key,value in compiled["resources"].items()}}
        return compiled,snapshot

    def _complete(self,record,files,details):
        descriptors = self.artifacts.write(record["id"],files)
        return self.catalog.transition(record["id"],{record["state"]},"complete",{**details,"artifacts":descriptors})

    def _fail(self,record,exc):
        detail = {"code":exc.code,"detail":exc.detail} if isinstance(exc,RenderError) else {"code":"operation_failed","detail":str(exc)}
        self.catalog.transition(record["id"],{record["state"]},"failed",{"error":detail})

    def render(self,request):
        parsed = RenderInput.model_validate(request).model_dump()
        with self.operation_lock:
            scene = self.catalog.get(parsed["scene"],"scene")
            require(scene["state"]=="active","scene_archived","scene is archived",409)
            compiled,snapshot = self._snapshot(scene)
            if parsed["stereo"] is not None:
                return self._render_stereo(parsed,scene,snapshot)
            record = self.catalog.create("render",{"request":parsed,"snapshot":snapshot},"queued")
            record = self.catalog.transition(record["id"],{"queued"},"running")
            try:
                started = time.perf_counter()
                buffers,metadata = render_samples(compiled,parsed)
                channels = resolve_samples(buffers,parsed["samples"])
                metadata.update({key:parsed[key] for key in ("width","height","samples")})
                metadata.update(node_map=compiled["node_map"],scene_digest=scene["digest"],
                                elapsed_seconds=time.perf_counter()-started)
                files = preview_channels(channels)
                files["channels.npz"] = packed_channels(channels)
                files["metadata.json"] = canonical(metadata).encode()
                return self._complete(record,files,{"metadata":metadata})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def _stereo_metadata(self,parsed,scene,rig_result,eyes,checks,analysis):
        report = rig_result["report"]
        left = eyes["left"]["metadata"]
        metadata = {
            "kind":"stereo-render",
            "primary_eye":"left",
            "request":{key:parsed[key] for key in ("width","height","samples","region")},
            "scene_id":scene["id"],"scene_digest":scene["digest"],
            "width":left["width"],"height":left["height"],"samples":left["samples"],
            "region":left["region"],
            "view":left["view"],"projection":left["projection"],
            "lens":left["lens"],
            "node_map":eyes["left"]["node_map"],
            "stats":{"left":eyes["left"]["stats"],"right":eyes["right"]["stats"]},
            "stereo":{**report,"checked":False},
            "units":{
                "world_unit":report["world_unit"],
                "scene_unit_meters":report["scene_unit_meters"],
                "interocular":{"value":report["interocular"]["value"],"unit":report["interocular"]["unit"],
                               "meters":report["interocular"]["meters"],"scene_units":report["interocular"]["scene_units"]},
                "convergence":{"value":report["convergence"]["value"],"unit":report["convergence"]["unit"],
                               "meters":report["convergence"]["meters"],"scene_units":report["convergence"]["scene_units"],
                               "source":report["convergence"]["source"],"measured_from":report["convergence"]["measured_from"]},
                "parallax":"full-frame pixels, right_x - left_x (negative crossed, positive uncrossed)",
                "vertical_parallax":"full-frame pixels, right_y - left_y (must stay 0 for a parallel rig)",
                "depth":"scene units along the rig forward axis from the baseline midpoint",
            },
            "rig_checks":checks,
            "eyes":{
                name:{
                    "eye_scene_units":eyes[name]["metadata"]["lens"]["eye"],
                    "shift_ndc":eyes[name]["metadata"]["lens"]["shift"],
                    "view":eyes[name]["view"],
                    "projection":eyes[name]["projection"],
                    "node_map":eyes[name]["node_map"],
                    "region":eyes[name]["metadata"]["region"],
                    "width":eyes[name]["metadata"]["width"],
                    "height":eyes[name]["metadata"]["height"],
                    "stats":eyes[name]["stats"],
                    "artifacts":["color.png","depth.png","normal.png","object-id.png","coverage.png","channels.npz"]
                        if name=="left" else ["right-color.png","right-channels.npz"],
                } for name in ("left","right")
            },
            "correspondence":{
                "query":"GET /v1/renders/{id}/correspondence/{side}/{x}/{y}",
                "status_legend":stereo_core.STATUS,
                "trust_rule":stereo_core.TRUST_RULE,
                "trusted_when":"status == 'matched'",
                "no_trusted_correspondence_when":"status in {'occluded_in_other_eye','surface_mismatch','no_destination_surface','outside_destination_frame','partial_coverage'} or the source pixel has no surface",
                "directions":{"left_to_right":analysis["summaries"]["left_to_right"],
                              "right_to_left":analysis["summaries"]["right_to_left"]},
                "probe_checks":analysis["probe_checks"],
                "artifact_semantics":{
                    "correspondence.png":"left-eye color with occluded or mismatched left pixels tinted red",
                    "occlusion-mask.png":"per-status color mask for the left-to-right direction",
                    "parallax.png":"red uncrossed / blue crossed horizontal parallax, grey untrusted",
                    "depth-probes.png":"left eye with numbered depth probes labelled measured/analytic parallax and depth",
                    "pair.png":"left and right eyes side by side with the explicit units in the header",
                    "anaglyph.png":"red channel from the left eye, green and blue from the right",
                    "correspondence.npz":"per-pixel status, parallax, vertical offset, destination pixel, depth and occluder for both directions",
                    "stereo.json":"rig, units and per-direction correspondence summary",
                    "crosscheck.json":"independent analytic probes and rig invariants",
                },
                "correspondence_npz_arrays":{
                    "L2R_status / R2L_status":"uint8 status code per source pixel (see status_legend)",
                    "L2R_parallax_pixels / R2L_parallax_pixels":"right_x - left_x, NaN when not trusted",
                    "L2R_vertical_pixels / R2L_vertical_pixels":"right_y - left_y, NaN when not trusted",
                    "L2R_destination_pixel / R2L_destination_pixel":"[x,y] in the other eye, -1 when off-frame",
                    "L2R_distance_scene_units / R2L_distance_scene_units":"depth along the rig forward axis",
                    "L2R_occluder_object_id / R2L_occluder_object_id":"node hit by the ray in the other eye when occluded",
                },
            },
        }
        metadata["stereo"]["checked"] = checks["translation_only"]
        metadata["stereo"]["focal_px"] = report["focal_px"]
        metadata["summary"] = _stereo_summary(metadata,checks,analysis)
        return metadata

    def _render_stereo(self,parsed,scene,snapshot):
        record = self.catalog.create("render",{"request":parsed,"snapshot":snapshot,"stereo":True},"queued")
        record = self.catalog.transition(record["id"],{"queued"},"running")
        try:
            started = time.perf_counter()
            rig_result = stereo_core.build_rig(snapshot["scene"]["lens"],parsed["stereo"],parsed["width"]/parsed["height"])
            eyes = stereo_core.render_stereo_eyes(snapshot,rig_result,parsed)
            checks = stereo_core.rig_checks(rig_result,eyes,parsed["width"],parsed["height"])
            analysis = stereo_core.analyze_stereo(eyes,rig_result)
            metadata = self._stereo_metadata(parsed,scene,rig_result,eyes,checks,analysis)
            files,preview = stereo_core.stereo_artifacts(eyes,analysis,rig_result)
            metadata["correspondence"]["parallax_preview_scale_pixels"] = preview["parallax_preview_scale_pixels"]
            metadata["correspondence"]["depth_probes"] = preview["depth_probe_legend"]
            metadata["elapsed_seconds"] = time.perf_counter()-started
            files.update(preview_channels(eyes["left"]["channels"]))
            files["channels.npz"] = packed_channels(eyes["left"]["channels"])
            files["metadata.json"] = canonical(metadata).encode()
            files["stereo.json"] = canonical({
                "format":"gnomon-stereo-metadata/1",
                "units":metadata["units"],
                "rig":metadata["stereo"],
                "rig_checks":checks,
                "correspondence":metadata["correspondence"],
            }).encode()
            files["crosscheck.json"] = canonical(
                stereo_core.crosscheck_material(rig_result,checks,analysis,eyes,metadata)).encode()
            descriptors = self.artifacts.write(record["id"],files)
            return self.catalog.transition(record["id"],{record["state"]},"complete",
                                           {"metadata":metadata,"artifacts":descriptors})
        except Exception as exc:
            self._fail(record,exc)
            raise

    def stereo_correspondence(self,identity,side,x,y):
        record = self.catalog.get(identity,"render")
        metadata = record["details"].get("metadata",{})
        require(metadata.get("kind")=="stereo-render","stereo_kind","render has no stereo pair",409)
        require(side in {"left","right"},"stereo_side","side must be left or right")
        rig = stereo_core.rig_from_metadata(metadata)
        opposite = "right" if side=="left" else "left"
        def channel_file(name):
            return "channels.npz" if name=="left" else "right-channels.npz"
        def eye(name):
            entry = metadata["eyes"][name]
            channels = unpack_channels(self.artifacts.read(record,channel_file(name)))
            return stereo_core.eye_arrays(
                channels,
                {"width":entry["width"],"height":entry["height"],"region":entry["region"],
                 "view":entry["view"],"projection":entry["projection"],
                 "lens":{"eye":entry["eye_scene_units"],"shift":entry["shift_ndc"]}},
                entry["node_map"])
        return stereo_core.correspondence_at(side,x,y,eye(side),eye(opposite),rig)

    def inspect(self,identity,x,y):
        record = self.catalog.get(identity,"render")
        channels = unpack_channels(self.artifacts.read(record,"channels.npz"))
        return inspect_pixel(channels,record["details"]["metadata"],x,y)

    def compare(self,request):
        parsed = CompareInput.model_validate(request).model_dump()
        with self.operation_lock:
            left = self.catalog.get(parsed["left"],"render")
            right = self.catalog.get(parsed["right"],"render")
            left_kind = left["details"].get("metadata",{}).get("kind")
            right_kind = right["details"].get("metadata",{}).get("kind")
            require((left_kind=="stereo-render")==(right_kind=="stereo-render"),"comparison_stereo",
                    "a stereo render can only be compared with another stereo render",409)
            basis = "primary_eye_left" if left_kind=="stereo-render" else "single_frame"
            record = self.catalog.create("comparison",parsed,"measuring")
            try:
                lc = unpack_channels(self.artifacts.read(left,"channels.npz"))
                rc = unpack_channels(self.artifacts.read(right,"channels.npz"))
                metrics,heat = compare_frames(lc,rc,left["details"]["metadata"],right["details"]["metadata"],parsed["threshold"])
                metrics["comparison_basis"] = basis
                return self._complete(record,{"difference.png":heat,"metrics.json":canonical(metrics).encode()},{"metrics":metrics})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def sheet(self,request):
        parsed = SheetInput.model_validate(request).model_dump()
        with self.operation_lock:
            scene = self.catalog.get(parsed["scene"],"scene")
            require(scene["state"]=="active","scene_archived","scene is archived",409)
            _,snapshot = self._snapshot(scene)
            record = self.catalog.create("sheet",{"request":parsed,"snapshot":snapshot},"building")
            try:
                def render_view(lens):
                    payload = {**snapshot["scene"],"lens":lens}
                    def frozen(identity,kind):
                        return snapshot["resources"][identity]
                    compiled = compile_scene(payload,frozen,allow_retired=True)
                    config = {"width":parsed["size"],"height":parsed["size"],"samples":1,"region":None}
                    buffers,metadata = render_samples(compiled,config)
                    channels = resolve_samples(buffers,1)
                    return preview_channels(channels)["color.png"],metadata
                png,index = build_sheet(snapshot["scene"],parsed["angles"],parsed["size"],parsed["columns"],render_view)
                return self._complete(record,{"sheet.png":png,"index.json":canonical(index).encode()},{"views":len(index),"index":index})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def bundle(self,identity):
        with self.operation_lock:
            render = self.catalog.get(identity,"render")
            require(render["state"]=="complete","bundle_incomplete","render must be complete",409)
            record = self.catalog.create("bundle",{"render":identity},"packing")
            try:
                files = {name:self.artifacts.read(render,name) for name in render["details"]["artifacts"]}
                raw,manifest = build_bundle(render,files)
                return self._complete(record,{"render.zip":raw},{"file_count":len(files),"manifest_format":manifest["format"]})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def record_summary(self,record):
        return {key:record[key] for key in ("id","kind","state","digest","details","created")}
