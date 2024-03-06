import pandas as pd
from pathlib import Path
import ruamel.yaml
import json
import numpy as np
import gc

# xsuite
import xtrack as xt
import xobjects as xo




#============================================================
def read_YAML(file="config.yaml"):
    ryaml = ruamel.yaml.YAML()
    # Read configuration for simulations
    with open(file, "r") as fid:
        config = ryaml.load(fid)

    return config
#============================================================

#============================================================
# Drop update configuration
def save_YAML(config,file="tmp_config.yaml"):
    ryaml = ruamel.yaml.YAML()
    with open(file, "w") as fid:
        ryaml.dump(config, fid)
#============================================================


#============================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
#============================================================


#============================================================
def mkdir(_path):
    if _path is not None:
        if isinstance(_path,Path):
            _path = str(_path)
        for parent in Path(_path+'/_').parents[::-1]:
            if parent.suffix=='':
                if not parent.exists():
                    parent.mkdir()    
#============================================================
                    


#============================================================
def whereis(obj: xo.HybridClass, _buffers=[]):
    context = obj._context.__class__.__name__
    if obj._buffer in _buffers:
        buffer_id = _buffers.index(obj._buffer)
    else:
        buffer_id = len(_buffers)
        _buffers.append(obj._buffer)
    offset = obj._offset
    print(f"context={context}, buffer={buffer_id}, offset={offset}")
#============================================================


#========================================
def parse_parquet_complex(A_vec):
    return [_A.view(dtype=np.complex128)[0] for _A in A_vec]

def import_parquet_datafile(path,partition_dict,columns = None,complex_columns = None,filters = None):
                   
    # -- DASK ----
    import dask.dataframe as dd
    import dask.config as ddconfig
    import gc
    ddconfig.set({"dataframe.convert-string": False})
    ddconfig.set({'dataframe.query-planning-warning': False})
    # https://dask.discourse.group/t/ddf-is-converting-column-of-lists-dicts-to-strings/2446
    #-------------



    # Setting up the filters
    #-----------------------------
    if filters is None:
        filters = [[]]
    
    partitionning = [(key, '==', val) for key,val in partition_dict.items()]
    filters = [f +partitionning for f in filters]

    assert ('data' in partition_dict.keys()), 'The partition_dict must contain data: ["naff","exursion","checkpoint","tbt"] to find datafile'
    parquet_file_extension = f'{partition_dict["data"]}_datafile_0.parquet'
    #-----------------------------
    
    # Importing the data
    #-----------------------------
    _partition = dd.read_parquet(path,columns=columns,filters = filters,parquet_file_extension = parquet_file_extension)
    df         = _partition.compute()
    #-----------------------------

    # Cleaning up the dataframe
    #-----------------------------
    if 'window' in df.columns:
        df = df[['window'] + [col for col in df.columns if col != 'window']]
    if 'chunk' in df.columns:
        df = df[['chunk'] + [col for col in df.columns if col != 'chunk']]
    
    df = df.reset_index(drop=True).drop(columns = list(partition_dict.keys()))
    #-----------------------------

    # Removing raw data
    #-----------------------------
    del(_partition)
    gc.collect()
    #-----------------------------

    # Parsing complex columns
    #-----------------------------
    if complex_columns is not None:
        for col in complex_columns:
            df[col] = df[col].apply(parse_parquet_complex)
    #-----------------------------
            
    return df
#========================================