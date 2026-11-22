import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from . import __version__
from .errors import RenderError
from .schema import MeshInput,TextureInput,SceneInput,RenderInput,CompareInput,SheetInput,LineLayerInput
from .service import RendererService


def create_app(root=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.renderer = RendererService(root or os.getenv("GNOMON_DATA",".gnomon-data"))
        yield
        app.state.renderer.close()
    app = FastAPI(title="Gnomon 静帧成像服务",version=__version__,lifespan=lifespan)
    @app.exception_handler(RenderError)
    async def domain_error(request,exc):
        return JSONResponse(status_code=exc.status,content={"code":exc.code,"detail":exc.detail})
    def service(request):
        return request.app.state.renderer
    @app.get("/health")
    def health(request:Request):
        return {"status":"ready","version":__version__,"backend":"numpy-cpu","recovered_operations":service(request).recovered_operations}
    @app.post("/v1/meshes",status_code=201)
    def meshes(payload:MeshInput,request:Request):
        return service(request).record_summary(service(request).add_mesh(payload.model_dump()))
    @app.post("/v1/textures",status_code=201)
    def textures(payload:TextureInput,request:Request):
        return service(request).record_summary(service(request).add_texture(payload.model_dump()))
    @app.post("/v1/scenes",status_code=201)
    def scenes(payload:SceneInput,request:Request):
        return service(request).record_summary(service(request).add_scene(payload.model_dump()))
    @app.post("/v1/scenes/{identity}/revisions",status_code=201)
    def revise(identity:str,payload:SceneInput,request:Request):
        return service(request).record_summary(service(request).add_scene(payload.model_dump(),identity))
    @app.post("/v1/scenes/{identity}/state/{target}")
    def scene_state(identity:str,target:str,request:Request):
        return service(request).record_summary(service(request).scene_state(identity,target))
    @app.post("/v1/resources/{identity}/state/{target}")
    def resource_state(identity:str,target:str,request:Request):
        return service(request).record_summary(service(request).resource_state(identity,target))
    @app.post("/v1/preflight")
    def preflight(payload:RenderInput,request:Request):
        return service(request).preflight(payload.model_dump())
    @app.post("/v1/renders",status_code=201)
    def render(payload:RenderInput,request:Request):
        return service(request).record_summary(service(request).render(payload.model_dump()))
    @app.get("/v1/renders/{identity}/pixels/{x}/{y}")
    def inspect(identity:str,x:int,y:int,request:Request):
        return service(request).inspect(identity,x,y)
    @app.post("/v1/comparisons",status_code=201)
    def compare(payload:CompareInput,request:Request):
        return service(request).record_summary(service(request).compare(payload.model_dump()))
    @app.post("/v1/sheets",status_code=201)
    def sheet(payload:SheetInput,request:Request):
        return service(request).record_summary(service(request).sheet(payload.model_dump()))
    @app.post("/v1/renders/{identity}/line-layers",status_code=201)
    def line_layer(identity:str,payload:LineLayerInput,request:Request):
        return service(request).record_summary(service(request).line_layer(identity,payload.model_dump()))
    @app.post("/v1/renders/{identity}/bundles",status_code=201)
    def bundle(identity:str,request:Request):
        return service(request).record_summary(service(request).bundle(identity))
    @app.get("/v1/records/{identity}")
    def get_record(identity:str,request:Request):
        return service(request).catalog.get(identity)
    @app.get("/v1/records")
    def list_records(kind:str,request:Request):
        from .errors import require
        require(kind in {"mesh","texture","scene","render","comparison","sheet","bundle","line_layer"},"record_kind","unknown record kind")
        return [service(request).record_summary(item) for item in service(request).catalog.list(kind)]
    @app.get("/v1/artifacts/{identity}/{name}")
    def artifact(identity:str,name:str,request:Request):
        renderer = service(request)
        raw = renderer.artifacts.read(renderer.catalog.get(identity),name)
        media = "image/png" if name.endswith(".png") else "application/zip" if name.endswith(".zip") else "application/json" if name.endswith(".json") else "application/octet-stream"
        return Response(raw,media_type=media,headers={"Content-Disposition":f'attachment; filename="{name}"'})
    return app


app = create_app()
