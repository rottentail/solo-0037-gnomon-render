import math
from io import BytesIO
from PIL import Image, ImageDraw
from .math3d import orbit_lens


def build_sheet(scene,angles,size,columns,render_view):
    rows = math.ceil(len(angles)/columns)
    canvas = Image.new("RGB",(columns*size,rows*(size+24)),(18,23,33))
    draw = ImageDraw.Draw(canvas)
    index = []
    for i,angle in enumerate(angles):
        lens = orbit_lens(scene["lens"],angle)
        raw,view_metadata = render_view(lens)
        x,y = (i%columns)*size,(i//columns)*(size+24)
        with Image.open(BytesIO(raw)) as image:
            canvas.paste(image,(x,y))
        draw.text((x+6,y+size+5),f"{i+1}: {angle:g} deg",fill=(211,224,238))
        index.append({"index":i+1,"angle":angle,"rect":[x,y,size,size],"lens":lens,"render":view_metadata})
    output = BytesIO()
    canvas.save(output,format="PNG")
    return output.getvalue(),index
