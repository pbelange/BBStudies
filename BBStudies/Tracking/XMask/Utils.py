
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



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def mkdir(_path):
    if _path is not None:
        for parent in Path(_path+'/_').parents[::-1]:
            if parent.suffix=='':
                if not parent.exists():
                    parent.mkdir()    


