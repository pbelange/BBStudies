
import json
import rich
import re
import numpy as np
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn,TimeElapsedColumn,SpinnerColumn,TimeRemainingColumn
import pickle
import dask.dataframe as dd
import gc
import traceback
# from pathlib import Path

import xobjects as xo
import xtrack as xt
import xpart as xp

from xdeps.refs import ARef

import BBStudies.Analysis.Footprint as footp




# ADDING FUNCTIONS TO AREF CLASS:
#============================================================
class RenderingKnobs(object):   
    def __init__(self, my_dict):
        for key in my_dict.keys():
            setattr(self, key, my_dict[key])


def knobs(self,render = True):
    _fields = self._value._fields
    
    sub_knobs   = []
    print_names = {}
    for key in _fields:

        _attr = getattr(self,key)

        # List or not list
        if isinstance(_attr._value, (type(np.array([])), list)):
            _expr = [_attr[i]._expr for i in range(len(_attr._value))]
        else:
            _expr = _attr._expr

        if _expr is None:
            print_names[key] = None
        else:
            print_names[key] = str(_expr)

        if str(_expr)[0] + str(_expr)[-1] == '[]':
            matches    = re.findall(r"[^[]*\[([^]]*)\]", str(_expr)[1:-1])
        else:
            matches    = re.findall(r"[^[]*\[([^]]*)\]", str(_expr))
        sub_knobs += [m[1:-1] for m in matches]

    print_values = {}
    for _var in list(set(sub_knobs)):
        _value = self._manager.containers['vars'][_var]._value
        print_values[f"'vars['{_var}']'"] = _value

    printable = {**print_values,**{ 30*'-': 30*'-'},**print_names}
    
    # Either shows the knobs or return list of knobs
    if render:
        rich.inspect(RenderingKnobs(printable),title=str(self._value), docs=False)
    else:
        return list(set(sub_knobs))


def inspect(self,**kwargs):
    return rich.inspect(self._value,**kwargs)

ARef.inspect = inspect
ARef.knobs   = knobs
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


# Loading line from file
#============================================================
def importLine(fname,force_energy = None):
    if force_energy is None:
        with open(fname, 'r') as fid:
            input_data = json.load(fid)
        line = xt.Line.from_dict(input_data)
        line.particle_ref = xp.Particles.from_dict(input_data['particle_on_tracker_co'])
    else:
        line = xt.Line.from_json(fname)
        line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=force_energy)
    return line
#============================================================


# Creating twiss b2 from b4
#==========================================
def twiss_b2_from_b4(twiss_b4):

    twiss_b2 = twiss_b4.copy()

    # Flipping x
    twiss_b2['x']   = -twiss_b2['x']

    # Need to flip py and dpy apparently?
    twiss_b2['py']  = -twiss_b2['py']
    twiss_b2['dpy'] = -twiss_b2['dpy']

    twiss_b2['dx']   = -twiss_b2['dx']
    twiss_b2['alfx'] = -twiss_b2['alfx']
    twiss_b2['alfy'] = -twiss_b2['alfy']

    twiss_b2['mux'] = np.max(twiss_b2['mux']) - twiss_b2['mux']
    twiss_b2['muy'] = np.max(twiss_b2['muy']) - twiss_b2['muy']

    # Flipping s
    lhcb2_L     = twiss_b2.loc['_end_point','s']
    twiss_b2['s'] = (-twiss_b2['s']+lhcb2_L).mod(lhcb2_L)
    twiss_b2.loc[['lhcb2ip3_p_','_end_point'],'s'] = lhcb2_L
    twiss_b2.sort_values(by='s',inplace=True)

    # Changing _den to _dex
    newIdx = twiss_b2.index.str.replace('_dex','_tmp_dex')
    newIdx = newIdx.str.replace('_den','_dex')
    newIdx = newIdx.str.replace('_tmp_dex','_den')
    twiss_b2.index = newIdx

    return twiss_b2
#==========================================


# Filtering twiss
#====================================
def filter_twiss(_twiss,entries = ['drift','..']):

    for ridof in entries:
        _twiss    =    _twiss[np.invert(_twiss.index.str.contains(ridof,regex=False))]

    return _twiss
#====================================


#====================================
def W_phys2norm(x,px,y,py,zeta,pzeta,W_matrix,particle_on_co,to_pd = False):
     

    # Compute ptau from delta
    #=======================================
    #beta0 = twiss.particle_on_co.beta0
    #delta_beta0 = delta * beta0
    #ptau_beta0 = (delta_beta0 * delta_beta0 + 2. * delta_beta0 * beta0 + 1.)**0.5 - 1.
    #ptau  = ptau_beta0 / beta0
    #pzeta = ptau / beta0
    #=======================================

    XX = np.zeros(shape=(6, len(x)), dtype=np.float64)
    XX[0,:] = x     - particle_on_co.x
    XX[1,:] = px    - particle_on_co.px
    XX[2,:] = y     - particle_on_co.y
    XX[3,:] = py    - particle_on_co.py
    XX[4,:] = zeta  - particle_on_co.zeta
    XX[5,:] = pzeta - particle_on_co.ptau / particle_on_co.beta0

    XX_n = np.dot(np.linalg.inv(W_matrix), XX)



    if to_pd:
        return pd.DataFrame({'x_n':XX_n[0,:],'px_n':XX_n[1,:],'y_n':XX_n[2,:],'py_n':XX_n[3,:],'zeta_n':XX_n[4,:],'pzeta_n':XX_n[5,:]})
    else:
        return XX_n

def norm2sigma(x_n,px_n,y_n,py_n,zeta_n,pzeta_n,nemitt_x,nemitt_y,nemitt_zeta,particle_on_co,to_pd = False):


    XX = np.zeros(shape=(6, len(x_n)), dtype=np.float64)
    XX[0,:] = x_n     / np.sqrt(nemitt_x/particle_on_co.gamma0)
    XX[1,:] = px_n    / np.sqrt(nemitt_x/particle_on_co.gamma0)
    XX[2,:] = y_n     / np.sqrt(nemitt_y/particle_on_co.gamma0)
    XX[3,:] = py_n    / np.sqrt(nemitt_y/particle_on_co.gamma0)
    XX[4,:] = zeta_n  / np.sqrt(nemitt_zeta/particle_on_co.gamma0)
    XX[5,:] = pzeta_n / np.sqrt(nemitt_zeta/particle_on_co.gamma0)

    if to_pd:
        return pd.DataFrame({'x_sig':XX[0,:],'px_sig':XX[1,:],'y_sig':XX[2,:],'py_sig':XX[3,:],'zeta_sig':XX[4,:],'pzeta_sig':XX[5,:]})
    else:
        return XX
#=======================================


#=======================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
#========================================


#========================================
def import_parquet(data_path,partition_name=None,partition_ID=None,variables = None):

    # Checking input
    #-----------------------------
    if variables is not None:
        if partition_name not in variables:
            variables = [partition_name] + variables
    #-----------------------------

    # Importing the data
    #-----------------------------
    if partition_ID is not None:
        assert (partition_name is None) == (partition_ID is None), 'partition_name and partition_ID must be both None or both not None'
        _partition = dd.read_parquet(data_path + f'/{partition_name}={partition_ID}',columns=variables,parquet_file_extension = '.parquet')
    else:
        _partition = dd.read_parquet(data_path,columns=variables,parquet_file_extension = '.parquet')
    #-----------------------------

    # Cleaning up the dataframe
    #-----------------------------
    df        = _partition.compute()
    if partition_name is not None:
        df = df.set_index(partition_name).reset_index()
    else:
        df = df.reset_index(drop=True)
    #-----------------------------


    # Removing raw data
    #-----------------------------
    del(_partition)
    gc.collect()
    #-----------------------------

    return df
#========================================


# def max_excursion(self):
    
#     x_max = np.max(self.monitor.x,axis=1)




# Calculation_Buffer:
#===================================================
class Calculation_Buffer():
    def __init__(self,monitor=None,user_calculations = None,user_args = None,_context=None):
        self.monitor           = monitor
        self.context           = _context
        self.user_calculations = user_calculations
        self.user_args         = user_args
        
        self.call_ID = None
        
        
        self.data = {}
        self.data['Chunk ID'] = []
        self.data['n_parts'] = []
        self.data['particle'] = []
        self.data['state']    = []
        self.data['start_at_turn'] = []
        self.data['stop_at_turn']  = []
        self.data['x_min'] = []
        self.data['x_max'] = []
        self.data['y_min'] = []
        self.data['y_max'] = []
        self.data['zeta_min'] = []
        self.data['zeta_max'] = []
        self.data['px_min'] = []
        self.data['px_max'] = []
        self.data['py_min'] = []
        self.data['py_max'] = []
        self.data['pzeta_min'] = []
        self.data['pzeta_max'] = []


    def to_pandas(self):
        dct = {}
        for key,value in self.data.items():
            if key == 'n_parts':
                continue
            
            # print(key,len(np.shape(value)))
            if len(np.shape(value)) == 1:
                dct[key] = np.repeat(value,self.data['n_parts'])
            elif len(np.shape(value)) == 2:
                dct[key] = np.hstack(value)

        return pd.DataFrame(dct)

    def process(self,monitor = None):

        # Initialize
        #-------------------------
        if self.call_ID is None:
            self.call_ID = 0
        else:
            self.call_ID += 1
        
        if monitor is not None:
            self.monitor = monitor
        #-------------------------

        start_at_turn = self.monitor.start_at_turn
        stop_at_turn  = self.monitor.stop_at_turn
        n_parts       = self.monitor.part_id_end - self.monitor.part_id_start

        # Note: 2D array are ordered following [particles,turns]
        x_max  = np.max(self.monitor.x,axis=1)
        px_max = np.max(self.monitor.px,axis=1)
        y_max  = np.max(self.monitor.y,axis=1)
        py_max = np.max(self.monitor.py,axis=1)
        zeta_max  = np.max(self.monitor.zeta,axis=1)

        # ptau/beta0 -> ignoring division by zeros
        pzeta = np.divide(self.monitor.ptau,self.monitor.beta0, np.zeros_like(self.monitor.ptau) + np.nan,where=self.monitor.beta0!=0)
        pzeta_max = np.max(pzeta,axis=1)

        # same for min
        x_min  = np.min(self.monitor.x,axis=1)
        px_min = np.min(self.monitor.px,axis=1)
        y_min  = np.min(self.monitor.y,axis=1)
        py_min = np.min(self.monitor.py,axis=1)
        zeta_min  = np.min(self.monitor.zeta,axis=1)
        pzeta_min = np.min(pzeta,axis=1)

        # Appending to data
        #-------------------------
        self.data['Chunk ID'].append(self.call_ID)
        self.data['n_parts'].append(n_parts)
        self.data['particle'].append(self.monitor.particle_id[:,-1])
        self.data['state'].append(self.monitor.state[:,-1])
        self.data['start_at_turn'].append(start_at_turn)
        self.data['stop_at_turn'].append(stop_at_turn)
        self.data['x_min'].append(x_min)
        self.data['x_max'].append(x_max)
        self.data['y_min'].append(y_min)
        self.data['y_max'].append(y_max)
        self.data['zeta_min'].append(zeta_min)
        self.data['zeta_max'].append(zeta_max)
        self.data['px_min'].append(px_min)
        self.data['px_max'].append(px_max)
        self.data['py_min'].append(py_min)
        self.data['py_max'].append(py_max)
        self.data['pzeta_min'].append(pzeta_min)
        self.data['pzeta_max'].append(pzeta_max)
        #-------------------------



def split_in_chunks(turns,n_chunks):
    main_chunk = turns//n_chunks
    chunks     = n_chunks*[main_chunk]+ [np.mod(turns,n_chunks)]
    if chunks[-1]==0:
        chunks = chunks[:-1]
    return chunks

# NEW Tracking class:
#===================================================
class Tracking_Interface():
    
    def __init__(self,line=None,particles=None,n_turns=None,method='6D',progress=False,progress_turn_chunk = None,_context=None,
                            monitor=None,rebuild = False,extract_columns = None,calculation_buffer = None,
                            nemitt_x = None,nemitt_y = None,nemitt_zeta = None,partition_name = None,partition_ID = None):
        
        # Tracking
        #-------------------------
        self.context        = _context
        self.context_name   = self.context.__class__.__name__
        self.monitor        = monitor
        self.partition_name = partition_name 
        self.partition_ID   = partition_ID 

        self.start_at_turn = None
        self.stop_at_turn  = None
        if n_turns is not None:
            self.n_turns   = int(n_turns)
        else:
            self.n_turns   = None
        if particles is not None:
            self.n_parts   = len(particles.particle_id)
        else:
            self.n_parts   = None
        
        # Saving emittance
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.nemitt_zeta = nemitt_zeta
        #-------------------------


        # Dataframes
        #-------------------------
        self._df       = None
        self._df_n     = None
        self._df_sig   = None

        self._coord    = None
        self._coord_n  = None
        self._coord_sig= None

        self._calculations = None
        self.calculation_buffer = calculation_buffer
        self.parquet_data  = '_df'

        if extract_columns is None:
            self.extract_columns = ['at_turn','particle_id','x','px','y','py','zeta','pzeta','state','at_element']
        #-------------------------


        # Footprint info
        #-------------------------
        self._tunes    = None
        self._tunes_n  = None
        self._tunesMTD    = 'nafflib'
        self._oldTunesMTD = 'nafflib'
        #-------------------------


        # Progress info
        #-------------------------
        self.progress_turn_chunk = progress_turn_chunk
        self.progress  = progress
        self.exec_time = None
        self._plive    = None
        self._pstatus  = None
        #-------------------------


        # Relevant twiss information
        #--------------------------
        if line is not None:
            _twiss = line.twiss(method=method.lower())
            self.W_matrix       = _twiss.W_matrix[0]
            self.particle_on_co = _twiss.particle_on_co 
        else:
            self.W_matrix       = None
            self.particle_on_co = None
        #--------------------------



        # Tracking
        #--------------------------
        if line is not None:
            self.method = method.lower()
            assert (method.lower() in ['4d','6d']), 'method should either be 4D or 6D (default)'
            try:
                if method=='4d':
                    line.freeze_longitudinal(True)

                # Track
                #=================
                self.runTracking(line,particles)
                #=================

                # Unfreeze longitudinal
                if method=='4d':
                    line.freeze_longitudinal(False)

            except Exception as error:
                self.closeLiveDisplay()
                print("An error occurred:", type(error).__name__, " - ", error)
                traceback.print_exc()
            except KeyboardInterrupt:
                self.closeLiveDisplay()
                print("Terminated by user: KeyboardInterrupt")
        #--------------------------

        # Disabling Tracking
        #-------------------------
        self.runTracking = lambda _: print('New Tracking instance needed')
        #-------------------------

    @classmethod
    def from_parquet(cls,data_path,partition_name=None,partition_ID=None,variables = None):
        self = cls()
        
        meta_path = f'{data_path}/{partition_name}={partition_ID}/meta_data.json'

        with open(meta_path , "r") as file: 
            metadata = json.load(file)

        self.parquet_data    = metadata['parquet_data']
        self.partition_name  = metadata['partition_name']
        self.partition_ID    = metadata['partition_ID']
        self.context_name    = metadata['context_name']
        self.exec_time       = metadata['exec_time']
        self.n_turns         = metadata['n_turns']
        self.start_at_turn   = metadata['start_at_turn']
        self.stop_at_turn    = metadata['stop_at_turn']
        self.n_parts         = metadata['n_parts']
        self.nemitt_x        = metadata['nemitt_x']
        self.nemitt_y        = metadata['nemitt_y']
        self.nemitt_zeta     = metadata['nemitt_zeta']
        self.method          = metadata['method']

        self.W_matrix        = np.array(metadata['W_matrix'])
        self.particle_on_co  = xp.Particles.from_dict(metadata['particle_on_co'])

        if self.parquet_data == '_df':
            self._df = import_parquet(data_path,partition_name=partition_name,partition_ID=partition_ID,variables = variables)
        elif self.parquet_data == '_calculations':
            self._calculations = import_parquet(data_path,partition_name=partition_name,partition_ID=partition_ID,variables = variables)

        return self


        
    
    def to_pickle(self,filename):
        self.context      = None
        self.progress     = None
        self.monitor      = None
        self.progress     = None
        self._plive       = None
        self._pstatus     = None
        self.runTracking  = None

        self._tunes    = None
        self._tunes_n  = None

        self._df       = None
        self._df_n     = None
        self._df_sig   = None

        self._coord    = None
        self._coord_n  = None
        self._coord_sig= None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def to_parquet(self,filename,partition_name = None,partition_ID = None,parquet_data = None):
        if partition_name is not None:
            self.partition_name = partition_name  
        if partition_ID is not None:
            self.partition_ID   = partition_ID
        if parquet_data is not None:
            self.parquet_data   = parquet_data

        # Export to parquet, partitioned in sub folder
        #---------------------------------------
        if self.parquet_data == '_df':
            _ = self.df
            self._df.insert(0,self.partition_name,self.partition_ID)
            self._df.to_parquet(filename,    partition_cols         = [self.partition_name],
                                            existing_data_behavior = 'delete_matching',
                                            basename_template      = 'tracking_data_{i}.parquet')
            self._df.drop(columns=[self.partition_name],inplace=True)
        elif self.parquet_data == '_calculations':
            _ = self.calculations
            self._calculations.insert(0,self.partition_name,self.partition_ID)
            self._calculations.to_parquet(filename, partition_cols         = [self.partition_name],
                                                    existing_data_behavior = 'delete_matching',
                                                    basename_template      = 'calculation_data_{i}.parquet')
            self._calculations.drop(columns=[self.partition_name],inplace=True)
        #---------------------------------------

        # Export metadata as well
        metadata = {'parquet_data'    : self.parquet_data,
                    'partition_name'  : self.partition_name,
                    'partition_ID'    : self.partition_ID,
                    'context_name'    : self.context_name,
                    'exec_time'       : self.exec_time,
                    'n_turns'         : self.n_turns,
                    'start_at_turn'   : self.start_at_turn,
                    'stop_at_turn'    : self.stop_at_turn,
                    'n_parts'         : self.n_parts,
                    'nemitt_x'        : self.nemitt_x,
                    'nemitt_y'        : self.nemitt_y,
                    'nemitt_zeta'     : self.nemitt_zeta,
                    'method'          : self.method,
                    'W_matrix'        : self.W_matrix,
                    'particle_on_co'  : self.particle_on_co.to_dict()}
        
        meta_path = f'{filename}/{self.partition_name}={self.partition_ID}/meta_data.json'
        with open(meta_path , "w") as outfile: 
            json.dump(metadata, outfile,cls=NpEncoder)

        

    @property
    def coord(self):
        if self._coord is None:
            self._coord = self.df.groupby('turn').get_group(0).drop(columns=['turn'])
        return self._coord
    
    @property
    def coord_n(self):
        if self._coord_n is None:
            self._coord_n = self.df_n.groupby('turn').get_group(0).drop(columns=['turn'])
        return self._coord_n
    
    @property
    def coord_sig(self):
        if self._coord_sig is None:
            self._coord_sig = self.df_sig.groupby('turn').get_group(0).drop(columns=['turn'])
        return self._coord_sig

    @property
    def calculations(self):
        if self._calculations is None:
            self._calculations = self.calculation_buffer.to_pandas()
        return self._calculations

    @property
    def df(self):
        if self._df is None:
            #CONVERT TO PANDAS
            self._df = pd.DataFrame(self.monitor.to_dict()['data'])
            
            # Getting rid of lost particles
            self._df = self._df[self._df['state'] != 0].reset_index(drop=True)

            # Filter the data
            self._df.insert(list(self._df.columns).index('zeta'),'pzeta',self._df['ptau']/self._df['beta0'])
            self._df = self._df[self.extract_columns]
            self._df.rename(columns={"at_turn": "turn",'particle_id':'particle'},inplace=True)

            # # Adding element name
            # if 'at_element' in self.extract_columns:
            #     self._df.loc[:,'at_element'] = self._df.at_element.apply(lambda ee_idx: line.element_names[ee_idx])
        
        return self._df

    @property
    def df_n(self):
        if self._df_n is None:
            coord_n = W_phys2norm(**self.df[['x','px','y','py','zeta','pzeta']],W_matrix=self.W_matrix,particle_on_co=self.particle_on_co,to_pd=True)
            self._df_n = pd.concat([self.df[['turn','particle']],coord_n],axis=1)
        
        return self._df_n
    

    @property
    def df_sig(self):
        if self._df_sig is None:
            # Asserting the existence of the emittances
            if (self.nemitt_x is None) or (self.nemitt_x is None) or (self.nemitt_zeta is None):
                print('Need to specifiy emittances, self.nemitt_x,self.nemitt_y,self.nemitt_zeta')
                return None
            
            # Computing in sigma coordinates
            coord_sig = norm2sigma(**self.df_n[['x_n','px_n','y_n','py_n','zeta_n','pzeta_n']],nemitt_x= self.nemitt_x, nemitt_y= self.nemitt_y, nemitt_zeta= self.nemitt_zeta, particle_on_co=self.particle_on_co,to_pd=True)
            self._df_sig = pd.concat([self.df_n[['turn','particle']],coord_sig],axis=1)
        
        return self._df_sig
    
    @property
    def tunes(self):
        # Reset if method is changed
        if self._tunesMTD != self._oldTunesMTD:
            self._tunes   = None
            self._tunes_n = None

        if self._tunes is None:
            if self._tunesMTD == 'pynaff':
                self._oldTunesMTD = 'pynaff'
                self._tunes    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.PyNAFF_tune(_part['x']),'Qy':footp.PyNAFF_tune(_part['y'])}))
            if self._tunesMTD == 'fft':
                self._oldTunesMTD = 'fft'
                self._tunes    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.FFT_tune(_part['x']),'Qy':footp.FFT_tune(_part['y'])}))
            if self._tunesMTD == 'nafflib':
                self._oldTunesMTD = 'nafflib'
                self._tunes    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.NAFFlib_tune(_part['x']),'Qy':footp.NAFFlib_tune(_part['y'])}))
        
        return self._tunes

    @property
    def tunes_n(self):
        # Reset if method is changed
        if self._tunesMTD != self._oldTunesMTD:
            self._tunes   = None
            self._tunes_n = None

        if self._tunes_n is None:
            if self._tunesMTD == 'pynaff':
                self._oldTunesMTD = 'pynaff'
                self._tunes_n    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.PyNAFF_tune(_part['x_n']),'Qy':footp.PyNAFF_tune(_part['y_n'])}))
            if self._tunesMTD == 'fft':
                self._oldTunesMTD = 'fft'
                self._tunes_n    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.FFT_tune(_part['x_n']),'Qy':footp.FFT_tune(_part['y_n'])}))
            if self._tunesMTD == 'nafflib':
                self._oldTunesMTD = 'nafflib'
                self._tunes_n    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.NAFFlib_tune(_part['x_n']),'Qy':footp.NAFFlib_tune(_part['y_n'])}))
        
        return self._tunes_n


    def initialize_monitor(self,start_at_turn=0,nturns = 1):
        self.monitor = xt.ParticlesMonitor( _context = self.context,num_particles = self.n_parts,
                                                start_at_turn    = start_at_turn, 
                                                stop_at_turn     = start_at_turn + nturns)

    
    def runTracking(self,line,particles):

        # Initiating monitor
        #-------------------------------
        if self.monitor is None:
            last_turn = self.context.nparray_from_context_array(particles.at_turn).max()
            self.initialize_monitor(start_at_turn=last_turn,nturns = self.n_turns)
        #-------------------------------

        # Saving turn infos
        self.start_at_turn = self.monitor.start_at_turn
        self.stop_at_turn  = self.monitor.stop_at_turn


        if not self.progress:
            # Regular tracking if no progress needed
            line.track(particles, num_turns=self.n_turns,turn_by_turn_monitor=self.monitor)

        else:
            
            # Splitting in desired progress chunk, or turn-by-turn
            # Note: there is always 1 single turn to start with to get a time estimate
            #-------------------------
            if self.progress_turn_chunk is None:
                self.progress_turn_chunk = 1
                
            main_chunk = self.n_turns-1
            chunks = [1] + (main_chunk//self.progress_turn_chunk)*[self.progress_turn_chunk]+ [np.mod(main_chunk,self.progress_turn_chunk)]
            #--------------------------

            # TRACKING
            self.startProgressBar()
            for chunk in chunks:
                if chunk == 0:
                    continue

                #---------------
                if self.calculation_buffer is not None:
                    # Overwrite monitor with every chunk
                    last_turn = self.context.nparray_from_context_array(particles.at_turn).max()
                    self.initialize_monitor(start_at_turn=last_turn,nturns = chunk)
                #---------------

                # Regular tracking with num_turns = chunk
                #---------------
                line.track(particles, num_turns=chunk,turn_by_turn_monitor=self.monitor)
                _ = self.monitor.stop_at_turn # Dummy access to data for time clock
                #---------------


                # Chunk Computation
                #---------------
                if self.calculation_buffer is not None:
                    self.calculation_buffer.process(monitor=self.monitor)
                #---------------
                

                self.updateProgressBar(chunk=chunk)


            #-------------------------
            self.closeLiveDisplay()



    # Progress bar methods
    #=============================================================================
    def startProgressBar(self,):
        self._plive = Progress("{task.description}",
                                TextColumn("[progress.remaining] ["),TimeRemainingColumn(),TextColumn("[progress.remaining]remaining ]   "),
                                SpinnerColumn(),
                                BarColumn(bar_width=40),
                                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                                TimeElapsedColumn())

        self._plive.start()
        self._plive.live._disable_redirect_io()

        self._pstatus = self._plive.add_task("[blue]Tracking\n", total=self.n_turns)
    
    def updateProgressBar(self,chunk = 1):
        self._plive.update(self._pstatus, advance=chunk,update=True)


    def startSpinner(self,):
        self._plive = Progress("{task.description}",
                                SpinnerColumn('aesthetic',),
                                TextColumn("[progress.elapsed] ["),TimeElapsedColumn (),TextColumn("[progress.elapsed]elapsed ]   "))

        self._plive.start()
        self._plive.live._disable_redirect_io()

        self._pstatus = self._plive.add_task("[blue]Tracking")


    def closeLiveDisplay(self,):
        self._plive.refresh()
        self._plive.stop()
        self._plive.console.clear_live()

        # Saving execution time in seconds
        self.exec_time = self._plive.tasks[0].finished_time

#===================================================







