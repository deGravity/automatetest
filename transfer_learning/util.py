import json
from zipfile import ZipFile
import os

def write_json(obj, filename):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(obj, f)

def load_json(p, zf=None):
    if zf is not None:
        if isinstance(zf, str):
            with ZipFile(zf, 'r') as z:
                with z.open(p,'r') as f:
                    return json.load(f)
        if isinstance(zf, ZipFile):
            with zf.open(p,'r') as f:
                return json.load(f)
    with open(p,'r') as f:
        return json.load(f)