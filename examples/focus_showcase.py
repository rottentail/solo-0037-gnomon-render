"""景深校样：同一静帧场景在针孔与三种清晰平面下的成像对比。

生成 focus-sheet.png（针孔 / 清晰平面在近景细条 / 中景主体 / 远景镂空纹理）
与 focus-edges.png（细条边缘与镂空纹理的三倍放大裁切，展示前景遮挡远景时
边缘的显露关系），并附 focus-showcase.json 记录每组清晰范围与采样代价。
"""
import json
import tempfile
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw
from gnomon.service import RendererService
from gnomon.samples import wedge_mesh, screen_mesh, slats_mesh, lattice_texture, depth_scene

VIEWS = [("pinhole", None, 1), ("near-slats", 5.8, 24), ("subject", 9.0, 24),
         ("far-lattice", 14.0, 24), ("far-lattice-x64", 14.0, 64)]
WIDTH, HEIGHT, SAMPLES, F_STOP = 320, 240, 2, 2.8
# 放大裁切区域（整图坐标）：细条伸出纹理上沿的边缘带，以及镂空纹理局部。
CROPS = {"slat-edge": (170, 0, 150, 60), "lattice": (30, 90, 110, 110)}


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-focus-") as root:
        service = RendererService(root)
        try:
            wedge = service.add_mesh(wedge_mesh())["id"]
            screen = service.add_mesh(screen_mesh())["id"]
            slats = service.add_mesh(slats_mesh())["id"]
            lattice = service.add_texture(lattice_texture())["id"]
            scene = service.add_scene(depth_scene(wedge, screen, slats, lattice))
            frames, records = {}, {}
            for name, distance, lens_samples in VIEWS:
                request = {"scene": scene["id"], "width": WIDTH, "height": HEIGHT, "samples": SAMPLES}
                if distance is not None:
                    request["focus"] = {"distance": distance, "f_stop": F_STOP, "samples": lens_samples}
                rendered = service.render(request)
                records[name] = rendered
                frames[name] = Image.open(BytesIO(service.artifacts.read(rendered, "color.png"))).convert("RGB")
            label_height = 20
            columns, rows = 2, 3
            sheet = Image.new("RGB", (columns * WIDTH, rows * (HEIGHT + label_height)), (18, 23, 33))
            draw = ImageDraw.Draw(sheet)
            for index, (name, _, _) in enumerate(VIEWS):
                x, y = (index % columns) * WIDTH, (index // columns) * (HEIGHT + label_height)
                sheet.paste(frames[name], (x, y))
                meta = records[name]["details"]["metadata"]
                focus = meta.get("focus")
                caption = f"{name}: pinhole" if focus is None else (
                    f"{name}: focus {focus['distance']} f/{focus['f_stop']} "
                    f"sharp [{focus['sharp_near']:.2f}, {focus['sharp_far']:.2f}] x{focus['samples']}")
                draw.text((x + 6, y + HEIGHT + 4), caption, fill=(211, 224, 238))
            sheet.save(output / "focus-sheet.png")
            zoom = 3
            edge = Image.new("RGB", (len(VIEWS) * CROPS["slat-edge"][2] * zoom,
                                     (CROPS["slat-edge"][3] + CROPS["lattice"][3]) * zoom + label_height), (18, 23, 33))
            edge_draw = ImageDraw.Draw(edge)
            for index, (name, _, _) in enumerate(VIEWS):
                x = index * CROPS["slat-edge"][2] * zoom
                y = 0
                for crop_name, (cx, cy, cw, ch) in CROPS.items():
                    piece = frames[name].crop((cx, cy, cx + cw, cy + ch)).resize((cw * zoom, ch * zoom), Image.NEAREST)
                    edge.paste(piece, (x, y))
                    y += ch * zoom
                edge_draw.text((x + 6, y + 3), name, fill=(211, 224, 238))
            edge.save(output / "focus-edges.png")
            summary = {name: {"request_focus": records[name]["details"]["metadata"].get("focus"),
                              "elapsed_seconds": records[name]["details"]["metadata"]["elapsed_seconds"],
                              "lens_passes": records[name]["details"]["metadata"]["stats"].get("lens_passes", 1)}
                       for name, _, _ in VIEWS}
            (output / "focus-showcase.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
            print(json.dumps({"output": str(output), "views": list(summary)}, ensure_ascii=False))
        finally:
            service.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run-output/focus-showcase"))
    produce(parser.parse_args().output)
