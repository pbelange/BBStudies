
from pathlib import Path
import ruamel.yaml
import json
import numpy as np

# xsuite
import xtrack as xt





def read_YAML(file="config.yaml"):
    ryaml = ruamel.yaml.YAML()
    # Read configuration for simulations
    with open(file, "r") as fid:
        config = ryaml.load(fid)

    return config

# Drop update configuration
def save_YAML(config,file="tmp_config.yaml"):
    ryaml = ruamel.yaml.YAML()
    with open(file, "w") as fid:
        ryaml.dump(config, fid)

# def read_multiline_and_config(file,keys = []):
#     # Loading attached configs
#     with open(file, "r") as fid:
#         content = json.load(fid)
#     configs = [content[key] for key in keys]

#     # Loading collider
#     _collider = xt.Multiline.from_json(file)

#     if len(configs)!=0:
#         return _collider,configs
#     else:
#         return _collider


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# def save_collider(_collider,file_path,_extra_dict=None):
#     if _extra_dict is not None:
#         collider_dict = _collider.to_dict()
#         for key,item in _extra_dict.items():
#             collider_dict[key] = item

#         with open(file_path, "w") as fid:
#             json.dump(collider_dict, fid, cls=NpEncoder)
        
#     else:
#         _collider.to_json(file_path)

def mkdir(_path):
    if _path is not None:
        for parent in Path(_path+'/_').parents[::-1]:
            if parent.suffix=='':
                if not parent.exists():
                    parent.mkdir()    


