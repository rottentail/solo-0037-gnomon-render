import hashlib
import os
from pathlib import Path
from .errors import require


class Artifacts:
    def __init__(self,root):
        self.root = Path(root)/"artifacts"
        self.root.mkdir(parents=True,exist_ok=True)

    def write(self,identity,files):
        folder = self.root/identity
        folder.mkdir(exist_ok=False)
        descriptors = {}
        try:
            for name,raw in files.items():
                require(Path(name).name==name,"artifact_name","artifact must have a simple name")
                target = folder/name
                temporary = folder/(name+".part")
                with temporary.open("wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(target)
                descriptors[name] = {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
        except Exception:
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()
            raise
        return descriptors

    def read(self,record,name):
        require(record["state"]=="complete","artifact_state","operation has no completed artifacts",409)
        descriptor = record["details"].get("artifacts",{}).get(name)
        require(descriptor is not None and Path(name).name==name,"artifact_missing","artifact not present",404)
        path = self.root/record["id"]/name
        require(path.is_file(),"artifact_lost","artifact is missing",409)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==descriptor["sha256"],"artifact_corrupt","artifact digest differs",409)
        return raw
