"""Keyframe-driven still-frame sequences.

Every frame is sampled at one shared time: parent and child node tracks advance
together, and the camera track uses the same clock. Rotation channels are
lifted onto a continuous degree branch first, so crossing the +/-180 degree
boundary follows the authored short turn instead of sweeping the long way
around. A frame rendered on its own uses exactly the same sampling path as the
batch, which makes single-frame re-renders byte-identical.
"""
import math
from copy import deepcopy
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw
from .errors import require
from .math3d import sample_channel, unwrap_degrees
from .scene import compile_scene
from .raster import render_samples
from .channels import resolve_samples, preview_channels, packed_channels


def schedule_times(request):
    if request["frame_times"] is not None:
        return list(request["frame_times"])
    return list(np.linspace(0.0, request["duration"], request["frame_count"]))


def _key_series(keyframes, field):
    """Return (times, raw values) for keyframes that define a given channel."""
    times, values = [], []
    for frame in keyframes:
        if frame[field] is not None:
            times.append(frame["time"])
            values.append(frame[field])
    return times, values


def _sample_vector(times, values, base, at, mode, rotate=False):
    result = []
    for axis in range(3):
        if len(times) < 2:
            result.append(float(base[axis]) if not times else float(values[0][axis]))
            continue
        axis_values = [item[axis] for item in values]
        if rotate:
            axis_values = unwrap_degrees(axis_values)
        result.append(sample_channel(times, axis_values, at, mode))
    return result


def sample_node_pose(track, base_pose, at):
    keyframes = track["keyframes"]
    mode = track["interpolation"]
    translate_t, translate_v = _key_series(keyframes, "translate")
    rotate_t, rotate_v = _key_series(keyframes, "rotate")
    scale_t, scale_v = _key_series(keyframes, "scale")
    return {
        "translate": tuple(_sample_vector(translate_t, translate_v, base_pose["translate"], at, mode)),
        "rotate": tuple(_sample_vector(rotate_t, rotate_v, base_pose["rotate"], at, mode, rotate=True)),
        "scale": tuple(_sample_vector(scale_t, scale_v, base_pose["scale"], at, mode)),
    }


def sample_lens(camera_track, base_lens, at):
    keyframes = camera_track["keyframes"]
    mode = camera_track["interpolation"]
    lens = deepcopy(base_lens)
    for field in ("eye", "target", "up"):
        times, values = _key_series(keyframes, field)
        lens[field] = tuple(_sample_vector(times, values, base_lens[field], at, mode))
    return lens


def validate_tracks(request, scene_payload, times):
    node_keys = {node["key"] for node in scene_payload["nodes"]}
    seen = set()
    for track in request["node_tracks"]:
        require(track["node"] in node_keys, "track_node", "track references an unknown scene node")
        require(track["node"] not in seen, "track_duplicate", "a node can have only one track")
        seen.add(track["node"])
        _validate_keyframe_times(track["keyframes"], request["duration"])
    require(request["node_tracks"] or request["camera_track"], "sequence_empty",
            "a sequence needs at least one node track or a camera track")
    if request["camera_track"]:
        _validate_keyframe_times(request["camera_track"]["keyframes"], request["duration"])
    # Pre-sample every frame so an out-of-bounds or singular pose fails before
    # any render record is created, rather than halfway through the sequence.
    base_poses = {node["key"]: node["pose"] for node in scene_payload["nodes"]}
    for at in times:
        for track in request["node_tracks"]:
            pose = sample_node_pose(track, base_poses[track["node"]], at)
            require(all(1e-6 <= abs(component) <= 10000 for component in pose["scale"]),
                    "track_scale", "interpolated scale must stay invertible and bounded")
            require(all(abs(component) <= 1e6 for component in pose["translate"]),
                    "track_translate", "interpolated translation outside supported coordinates")
        if request["camera_track"]:
            lens = sample_lens(request["camera_track"], scene_payload["lens"], at)
            require(np.linalg.norm(lens["up"]) > 1e-12, "track_camera", "interpolated camera up has zero length")
            require(all(abs(v) <= 1e6 for vector in (lens["eye"], lens["target"]) for v in vector),
                    "track_camera", "interpolated camera coordinates outside supported range")


def _validate_keyframe_times(keyframes, duration):
    times = [frame["time"] for frame in keyframes]
    require(all(times[i] < times[i + 1] for i in range(len(times) - 1)),
            "track_keyframes", "keyframe times must be strictly increasing")
    require(0.0 <= times[0] and times[-1] <= duration,
            "track_keyframes", "keyframe times must lie within [0, duration]")


def frame_specs(request, scene_payload, times):
    """All per-frame inputs in frame order. Pure data: no rendering here."""
    base_poses = {node["key"]: node["pose"] for node in scene_payload["nodes"]}
    specs = []
    for index, at in enumerate(times):
        poses = {track["node"]: sample_node_pose(track, base_poses[track["node"]], at)
                 for track in request["node_tracks"]}
        lens = (sample_lens(request["camera_track"], scene_payload["lens"], at)
                if request["camera_track"] else deepcopy(scene_payload["lens"]))
        specs.append({"index": index, "time": at, "poses": poses, "lens": lens})
    return specs


def _scene_at(snapshot_scene, spec):
    payload = deepcopy(snapshot_scene)
    for node in payload["nodes"]:
        if node["key"] in spec["poses"]:
            node["pose"] = deepcopy(spec["poses"][node["key"]])
    payload["lens"] = deepcopy(spec["lens"])
    return payload


def render_frame(snapshot, request_config, spec):
    """Render one frame from a frozen scene/resource snapshot."""
    payload = _scene_at(snapshot["scene"], spec)

    def frozen(identity, kind):
        return snapshot["resources"][identity]

    compiled = compile_scene(payload, frozen, allow_retired=True)
    config = {"width": request_config["width"], "height": request_config["height"],
              "samples": request_config["samples"], "region": None}
    buffers, metadata = render_samples(compiled, config)
    channels = resolve_samples(buffers, config["samples"])
    files = preview_channels(channels)
    files["channels.npz"] = packed_channels(channels)
    return files, channels, metadata


def build_contact_sheet(png_list, times, width, height, columns):
    rows = math.ceil(len(png_list) / columns)
    label_height = 22
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), (18, 23, 33))
    draw = ImageDraw.Draw(canvas)
    for index, raw in enumerate(png_list):
        x, y = (index % columns) * width, (index // columns) * (height + label_height)
        with Image.open(BytesIO(raw)) as image:
            canvas.paste(image, (x, y))
        draw.text((x + 6, y + height + 5), f"f{index} t={times[index]:.3f}s", fill=(211, 224, 238))
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def build_sequence(snapshot, request, specs):
    """Render every spec through the shared single-frame path."""
    frames, color_arrays = [], []
    files = {}
    for spec in specs:
        frame_files, channels, metadata = render_frame(snapshot, request, spec)
        name = f"frame-{spec['index']:03d}"
        for artifact, raw in frame_files.items():
            files[f"{name}-{artifact}"] = raw
        color_arrays.append(np.asarray(channels["color"]))
        frames.append({
            "index": spec["index"],
            "time": spec["time"],
            "inputs": {"node_poses": spec["poses"], "lens": spec["lens"]},
            "artifacts": {artifact: f"{name}-{artifact}" for artifact in sorted(frame_files)},
            "render": {"stats": metadata["stats"], "region": metadata["region"]},
        })
    files["contact-sheet.png"] = build_contact_sheet(
        [files[frame["artifacts"]["color.png"]] for frame in frames],
        [f["time"] for f in frames],
        request["width"], request["height"], request["columns"])
    stacked = BytesIO()
    np.savez_compressed(stacked, color=np.stack(color_arrays),
                        time=np.asarray([f["time"] for f in frames]))
    files["frames.npz"] = stacked.getvalue()
    manifest = {
        "format": "gnomon-sequence/1",
        "scene_id": snapshot["scene_id"],
        "scene_digest": snapshot["scene_digest"],
        "schedule": {"duration": request["duration"],
                     "frame_times": [spec["time"] for spec in specs]},
        "output": {"width": request["width"], "height": request["height"], "samples": request["samples"]},
        "interpolation_modes": {
            track["node"]: track["interpolation"] for track in request["node_tracks"]
        } | ({"camera": request["camera_track"]["interpolation"]} if request["camera_track"] else {}),
        "tracks": {"node_tracks": request["node_tracks"],
                   "camera_track": request["camera_track"]},
        "frames": frames,
    }
    return files, manifest
