
import json
import rich
import re
import numpy as np
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn,TimeElapsedColumn,SpinnerColumn,TimeRemainingColumn
import pickle

import gc
import traceback
from pathlib import Path
import nafflib

import xobjects as xo
import xtrack as xt
import xpart as xp



import BBStudies.Analysis.Footprint as footp
import BBStudies.Tracking.Progress as pbar
import BBStudies.Physics.Constants as cst


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
class RFBucket(xp.longitudinal.rf_bucket.RFBucket):
    # From https://github.com/xsuite/xpart/blob/main/xpart/longitudinal/rf_bucket.py
    def __init__(self,line):
        dct_longitudinal = xp.longitudinal.generate_longitudinal._characterize_line(line,line.particle_ref)
        dct_longitudinal['circumference'] = line.get_length()
        dct_longitudinal['gamma'] = line.particle_ref.gamma0[0]
        dct_longitudinal['mass_kg'] = line.particle_ref.mass0/(cst.c**2)*cst.elec
        dct_longitudinal['charge_coulomb'] = np.abs(line.particle_ref.q0)*cst.elec
        dct_longitudinal['momentum_compaction_factor'] = line.twiss()['momentum_compaction_factor']

        super().__init__(circumference      = dct_longitudinal['circumference'],
                            gamma           = dct_longitudinal['gamma'],
                            mass_kg         = dct_longitudinal['mass_kg'],
                            charge_coulomb  = dct_longitudinal['charge_coulomb'],
                            alpha_array     = np.atleast_1d(dct_longitudinal['momentum_compaction_factor']),
                            harmonic_list   = np.atleast_1d(dct_longitudinal['h_list']),
                            voltage_list    = np.atleast_1d(dct_longitudinal['voltage_list']),
                            phi_offset_list = np.atleast_1d((np.array(dct_longitudinal['lag_list_deg']) - 180)/180*np.pi),
                            p_increment     = 0)

    @property
    def zeta_max(self):
        return self.circumference / (2*np.amin(self.h))

    def invariant(self,zeta0,npoints = 1000):
        # Returns the positive branch of the invariant  crossing (zeta,delta) = (zeta0,0)
        zeta_vec  = np.linspace(-zeta0,zeta0,npoints)
        delta_vec = self.equihamiltonian(zcut=zeta0)(zeta_vec)
        return zeta_vec,delta_vec
    
    
    def compute_emittance(self,sigma_z = 0.09):
        matcher = xp.longitudinal.rfbucket_matching.RFBucketMatcher(rfbucket            = self, 
                                                                    distribution_type   = xp.longitudinal.rfbucket_matching.ThermalDistribution,
                                                                    sigma_z             = sigma_z)
        _,_, _, _ = matcher.generate(macroparticlenumber=1)
        return matcher._compute_emittance(matcher.rfbucket,matcher.psi)
#====================================


#====================================
def delta2pzeta(delta,beta0):
    # from https://github.com/xsuite/xpart/blob/a1232d03fc0ee90cb4b64fe9f0ce086c68934f5a/xpart/build_particles.py#L134C1-L143C43
    delta_beta0 = delta * beta0
    ptau_beta0 = (delta_beta0 * delta_beta0
                        + 2. * delta_beta0 * beta0 + 1.)**0.5 - 1.
    pzeta = ptau_beta0 / beta0 / beta0
    return pzeta
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

    gamma0 = particle_on_co.gamma0[0]
    XX = np.zeros(shape=(6, len(x_n)), dtype=np.float64)
    XX[0,:] = x_n     / np.sqrt(nemitt_x/gamma0)
    XX[1,:] = px_n    / np.sqrt(nemitt_x/gamma0)
    XX[2,:] = y_n     / np.sqrt(nemitt_y/gamma0)
    XX[3,:] = py_n    / np.sqrt(nemitt_y/gamma0)
    XX[4,:] = zeta_n  / np.sqrt(nemitt_zeta/gamma0)
    XX[5,:] = pzeta_n / np.sqrt(nemitt_zeta/gamma0)

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
def import_parquet(data_path,partition_name=None,partition_ID=None,variables = None,start_at_turn = None,stop_at_turn = None,handpick_particles = None):

    # -- DASK ----
    import dask.dataframe as dd
    import dask.config as ddconfig
    ddconfig.set({"dataframe.convert-string": False})
    # https://dask.discourse.group/t/ddf-is-converting-column-of-lists-dicts-to-strings/2446
    #-------------



    # Checking input
    #-----------------------------
    if variables is not None:
        if partition_name not in variables:
            variables = [partition_name] + variables

    filters = None
    if (start_at_turn is not None) or (stop_at_turn is not None):
        if start_at_turn is None:
            start_at_turn = 0
        if stop_at_turn is None:
            stop_at_turn = 1e10
        
        filters = [[('turn','>=',start_at_turn),('turn','<=',stop_at_turn)]]
    
    if handpick_particles is not None:
        if filters is None:
            filters = [[]]
        filters = [filters[0]+[('particle','==',part)] for part in handpick_particles]
    #-----------------------------

    # Importing the data
    #-----------------------------
    if partition_ID is not None:
        assert (partition_name is None) == (partition_ID is None), 'partition_name and partition_ID must be both None or both not None'
        _partition = dd.read_parquet(data_path + f'/{partition_name}={partition_ID}',columns=variables,filters = filters,parquet_file_extension = '.parquet')
    else:
        _partition = dd.read_parquet(data_path,columns=variables,filters = filters,parquet_file_extension = '.parquet')
    #-----------------------------

    # Cleaning up the dataframe
    #-----------------------------
    df        = _partition.compute()
    if partition_name is not None:
        df = df.set_index(partition_name).reset_index(drop=True)
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


#===================================================
class Checkpoint_Buffer():
    def __init__(self,monitor=None):
        self.monitor = monitor
        self.call_ID = None
        
        self.data = {}
        self.data['Chunk ID'] = []
        self.data['turn']     = []
        self.data['particle'] = []
        self.data['state']    = []
        self.data['x']        = []
        self.data['px']       = []
        self.data['y']        = []
        self.data['py']       = []
        self.data['zeta']     = []
        self.data['pzeta']    = []
        
        self.particle_id = None

    def to_pandas(self):
        dct    = {}
        nparts = len(self.particle_id)

        for key,value in self.data.items():

            if len(np.shape(value)) == 1:
                dct[key] = np.repeat(value,nparts)
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

        if self.particle_id is None:
            self.particle_id = np.arange(self.monitor.part_id_start,self.monitor.part_id_end)
        #-------------------------

        start_at_turn = self.monitor.start_at_turn

        # ptau/beta0 -> ignoring division by zeros
        pzeta = np.divide(self.monitor.ptau,self.monitor.beta0, np.zeros_like(self.monitor.ptau) + np.nan,where=self.monitor.beta0!=0)

        # Appending to data
        #-------------------------
        self.data['Chunk ID'].append(self.call_ID)
        self.data['turn'].append(start_at_turn)
        self.data['particle'].append(self.particle_id)
        self.data['state'].append(self.monitor.state[:,0])
        self.data['x'].append(self.monitor.x[:,0])
        self.data['px'].append(self.monitor.px[:,0])
        self.data['y'].append(self.monitor.y[:,0])
        self.data['py'].append(self.monitor.py[:,0])
        self.data['zeta'].append(self.monitor.zeta[:,0])
        self.data['pzeta'].append(pzeta[:,0])
        #-------------------------

# Excursion_Buffer:
#===================================================
class Excursion_Buffer():
    def __init__(self,monitor=None):
        self.monitor = monitor
        self.call_ID = None
        
        
        self.data = {}
        self.data['Chunk ID'] = []
        self.data['particle'] = []
        self.data['state']    = []
        self.data['start_at_turn'] = []
        self.data['stop_at_turn']  = []
        self.data['x_min'] = []
        self.data['x_max'] = []
        self.data['y_min'] = []
        self.data['y_max'] = []
        self.data['skew_min'] = []
        self.data['skew_max'] = []
        self.data['px_min'] = []
        self.data['px_max'] = []
        self.data['py_min'] = []
        self.data['py_max'] = []
        self.data['Qx'] = []
        self.data['Qy'] = []
        self.data['Qzeta'] = []


        self.particle_id = None


    def to_pandas(self):
        dct    = {}
        nparts = len(self.particle_id)

        for key,value in self.data.items():

            if len(np.shape(value)) == 1:
                dct[key] = np.repeat(value,nparts)
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

        if self.particle_id is None:
            self.particle_id = np.arange(self.monitor.part_id_start,self.monitor.part_id_end)
        #-------------------------


        # Extracting data
        #-------------------------
        start_at_turn = self.monitor.start_at_turn
        stop_at_turn  = self.monitor.stop_at_turn

        x    = self.monitor.x
        px   = self.monitor.px
        y    = self.monitor.y
        py   = self.monitor.py
        zeta = self.monitor.zeta
        pzeta = np.divide(self.monitor.ptau,self.monitor.beta0, np.zeros_like(self.monitor.ptau) + np.nan,where=self.monitor.beta0!=0)
         
        # Rotating for skew collimator
        #-------------------------
        skew_angle   = 127.5 + 90 #skew coll angle, +90 because 0 corresponds to a horizontal collimator_n/vertical walls
        theta_unskew = -np.deg2rad(skew_angle-90)
        x_skew       = x*np.cos(theta_unskew) - y*np.sin(theta_unskew)

        # Tunes
        #-------------------------
        Qx    = nafflib.tune(x, px, window_order=2, window_type="hann")
        Qy    = nafflib.tune(y, py, window_order=2, window_type="hann")
        Qzeta = nafflib.tune(zeta, pzeta, window_order=2, window_type="hann")

        # Extracting max and min || Note: 2D array are ordered following [particles,turns]
        #-------------------------
        # X -------------
        idx_list = np.arange(len(self.particle_id))
        idx_max  = np.argmax(x,axis=1)
        idx_min  = np.argmin(x,axis=1)
        x_max,px_max = x[idx_list,idx_max],self.monitor.px[idx_list,idx_max]
        x_min,px_min = x[idx_list,idx_min],self.monitor.px[idx_list,idx_min]

        # Y -------------
        idx_max = np.argmax(y,axis=1)
        idx_min = np.argmin(y,axis=1)
        y_max,py_max = y[idx_list,idx_max],self.monitor.py[idx_list,idx_max]
        y_min,py_min = y[idx_list,idx_min],self.monitor.py[idx_list,idx_min]


        # Skew ----------
        skew_max = np.max(x_skew,axis=1)
        skew_min = np.min(x_skew,axis=1)


        # Appending to data
        #-------------------------
        self.data['Chunk ID'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['state'].append(self.monitor.state[:,-1])
        self.data['start_at_turn'].append(start_at_turn)
        self.data['stop_at_turn'].append(stop_at_turn)
        self.data['x_min'].append(x_min)
        self.data['x_max'].append(x_max)
        self.data['y_min'].append(y_min)
        self.data['y_max'].append(y_max)
        self.data['skew_min'].append(skew_min)
        self.data['skew_max'].append(skew_max)
        self.data['px_min'].append(px_min)
        self.data['px_max'].append(px_max)
        self.data['py_min'].append(py_min)
        self.data['py_max'].append(py_max)
        self.data['Qx'].append(Qx)
        self.data['Qy'].append(Qy)
        self.data['Qzeta'].append(Qzeta)
        #-------------------------


# naff_Buffer:
#===================================================
class naff_Buffer():
    def __init__(self,monitor=None):
        self.monitor = monitor
        self.call_ID = None
        
        
        self.data = {}
        self.data['Chunk ID'] = []
        self.data['particle'] = []
        self.data['state']    = []
        self.data['start_at_turn'] = []
        self.data['stop_at_turn']  = []

        self.data['Ax']  = []
        self.data['Qx']  = []
        self.data['Ay']  = []
        self.data['Qy']  = []
        self.data['Azeta']  = []
        self.data['Qzeta']  = []


        self.particle_id = None




    def to_dict(self):
        dct    = {}
        nparts = len(self.particle_id)

        for key,value in self.data.items():

            if len(np.shape(value)) == 1:
                dct[key] = np.repeat(value,nparts)
            elif len(np.shape(value)) == 2:
                dct[key] = np.hstack(value)
            elif len(np.shape(value)) == 3:
                # numpy array for each particle
                if np.issubdtype(value[0].dtype, complex):
                    # is complex
                    dct[key] = [[(c.real, c.imag) for c in row] for row in np.vstack(value).tolist()]
                else:
                    dct[key] = np.vstack(value).tolist()

        return dct
    
    def to_pandas(self):
        return pd.DataFrame(self.to_dict())

    def process(self,monitor = None):

        # Initialize
        #-------------------------
        if self.call_ID is None:
            self.call_ID = 0
        else:
            self.call_ID += 1
        
        if monitor is not None:
            self.monitor = monitor

        if self.particle_id is None:
            self.particle_id = np.arange(self.monitor.part_id_start,self.monitor.part_id_end)
        #-------------------------


        # Extracting data
        #-------------------------
        start_at_turn = self.monitor.start_at_turn
        stop_at_turn  = self.monitor.stop_at_turn

        x    = self.monitor.x
        px   = self.monitor.px
        y    = self.monitor.y
        py   = self.monitor.py
        zeta = self.monitor.zeta
        pzeta = np.divide(self.monitor.ptau,self.monitor.beta0, np.zeros_like(self.monitor.ptau) + np.nan,where=self.monitor.beta0!=0)
         
        # Extracting 10 harmonics
        #--------------------------
        n_harm = 10
        window_order = 4
        window_type  = 'hann' 
        try:
            Ax,Qx  = nafflib.multiparticle_harmonics(x, px, num_harmonics=n_harm, window_order=window_order, window_type=window_type)
            Ay,Qy  = nafflib.multiparticle_harmonics(y, py, num_harmonics=n_harm, window_order=window_order, window_type=window_type)
            Azeta,Qzeta  = nafflib.multiparticle_harmonics(zeta, pzeta, num_harmonics=n_harm, window_order=window_order, window_type=window_type)
        except Exception as error:
            print("An exception occurred:", type(error).__name__, "-", error) # An exception occurred
            n_part = len(x)
            Ax,Qx = n_part * [n_harm*[np.nan+ 1j*np.nan]],n_part * [n_harm*[np.nan]]
            Ay,Qy =  n_part * [n_harm*[np.nan+ 1j*np.nan]],n_part * [n_harm*[np.nan]]
            Azeta,Qzeta =  n_part * [n_harm*[np.nan+ 1j*np.nan]],n_part * [n_harm*[np.nan]]


        # Appending to data
        #-------------------------
        self.data['Chunk ID'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['state'].append(self.monitor.state[:,-1])
        self.data['start_at_turn'].append(start_at_turn)
        self.data['stop_at_turn'].append(stop_at_turn)
        #----------
        self.data['Ax'].append(Ax)
        self.data['Qx'].append(Qx)
        self.data['Ay'].append(Ay)
        self.data['Qy'].append(Qy)
        self.data['Azeta'].append(Azeta)
        self.data['Qzeta'].append(Qzeta)
        #-------------------------

def split_in_chunks(turns,n_chunks = None,main_chunk = None):
    if n_chunks is not None:
        # See https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split
        l = turns
        n = n_chunks
        chunks = (l % n) * [l//n + 1] + (n-(l % n))*[l//n]
        
    elif main_chunk is not None:
        n_chunks = turns//main_chunk
        chunks   = n_chunks*[main_chunk]+ [np.mod(turns,main_chunk)]
    
    if chunks[-1]==0:
        chunks = chunks[:-1]

    return chunks


class coordinate_table():
    def __init__(self,_df,W_matrix=None,particle_on_co=None,nemit_x=None,nemit_y=None,nemit_zeta=None):
        self._df     = _df
        self._df_n   = None
        self._df_sig = None

        self.W_matrix = W_matrix
        self.particle_on_co = particle_on_co
        self.nemitt_x = nemit_x
        self.nemitt_y = nemit_y
        self.nemitt_zeta = nemit_zeta

    @property
    def df(self):
        return self._df

    @property
    def df_n(self):
        if self._df_n is None:
            coord_n    = W_phys2norm(**self.df[['x','px','y','py','zeta','pzeta']],W_matrix=self.W_matrix,particle_on_co=self.particle_on_co,to_pd=True)
            old_cols   = list(self.df.columns.drop(['x','px','y','py','zeta','pzeta']))
            self._df_n = pd.concat([self.df[old_cols],coord_n],axis=1)
        return self._df_n
    

    @property
    def df_sig(self):
        if self._df_sig is None:
            # Asserting the existence of the emittances
            if (self.nemitt_x is None) or (self.nemitt_x is None) or (self.nemitt_zeta is None):
                print('Need to specifiy emittances, self.nemitt_x,self.nemitt_y,self.nemitt_zeta')
                return None
            
            # Computing in sigma coordinates
            coord_sig    = norm2sigma(**self.df_n[['x_n','px_n','y_n','py_n','zeta_n','pzeta_n']],nemitt_x= self.nemitt_x, nemitt_y= self.nemitt_y, nemitt_zeta= self.nemitt_zeta, particle_on_co=self.particle_on_co,to_pd=True)
            old_cols     = list(self.df_n.columns.drop(['x_n','px_n','y_n','py_n','zeta_n','pzeta_n']))
            self._df_sig = pd.concat([self.df_n[old_cols],coord_sig],axis=1)
        return self._df_sig



# NEW Tracking class:
#===================================================
class Tracking_Interface():
    
    def __init__(self,line=None,particles=None,n_turns=None,method='6D',Pbar = None,progress=False,progress_divide = 100,_context=None,
                            monitor=None,monitor_at = None,extract_columns = None,
                            nemitt_x = None,nemitt_y = None,nemitt_zeta = None,sigma_z = None,partition_name = None,partition_ID = None,config=None):
        
        # Tracking
        #-------------------------
        self.context        = _context
        self.context_name   = self.context.__class__.__name__
        self.monitor        = monitor
        self.partition_name = partition_name 
        self.partition_ID   = partition_ID 
        self.config         = config

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
        self.nemitt_x    = nemitt_x
        self.nemitt_y    = nemitt_y
        self.nemitt_zeta = nemitt_zeta
        self.sigma_z  = sigma_z
        #-------------------------


        # Dataframes
        #-------------------------
        self._df       = None
        self._coord    = None
        self._checkpoint = None

        self._data = None
        
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
        self.progress_divide = progress_divide
        self.progress        = progress
        if (Pbar is None) and (progress):
            self.PBar = pbar.ProgressBar(message = 'Tracking ...',color='blue',n_steps = self.n_turns)
        elif Pbar is not None:
            self.PBar     = Pbar
            self.progress = True
        else:
            self.PBar     = None
            self.progress = False
        self.exec_time = None
        #-------------------------


        # Relevant twiss information
        #--------------------------

        # cycle if needed
        #--------
        self.monitor_at = monitor_at
        if self.monitor_at is not None:
            if line.element_names[0] != self.monitor_at:
                at_element_idx = line.element_names.index(self.monitor_at)
                assert particles.at_element == at_element_idx, 'particles should be generated at the lcoation of the monitor'
                # Cycling
                print(f'__ CYCLING LINE AT {self.monitor_at} __')
                line.cycle(name_first_element=self.monitor_at, inplace=True)
                # Setting at_element accordingly
                particles.at_element               *= 0  
                particles.start_tracking_at_element = 0
        #--------
        
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
                self.run_tracking(line,particles)
                #=================

                # Unfreeze longitudinal
                if method=='4d':
                    line.freeze_longitudinal(False)

            except Exception as error:
                self.PBar.close()
                print("An error occurred:", type(error).__name__, " - ", error)
                traceback.print_exc()
            except KeyboardInterrupt:
                self.PBar.close()
                print("Terminated by user: KeyboardInterrupt")
        #--------------------------

        # Disabling Tracking
        #-------------------------
        self.run_tracking = lambda _: print('New Tracking instance needed')
        #-------------------------


    def to_dict(self):
        metadata = {'config'          : self.config,
                    'parquet_data'    : self.parquet_data,
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
                    'sigma_z'         : self.sigma_z,
                    'method'          : self.method,
                    'monitor_at'      : self.monitor_at,
                    'W_matrix'        : self.W_matrix,
                    'particle_on_co'  : self.particle_on_co.to_dict()}
        return metadata
    

    @classmethod
    def from_parquet(cls,data_path,partition_name=None,partition_ID=None,variables = None,start_at_turn = None,stop_at_turn = None,handpick_particles = None):
        self = cls()
        
        # Extracting metadata
        #-------------------------
        if partition_ID is not None:
            meta_path = f'{data_path}/{partition_name}={partition_ID}/meta_data.json'
        else:
            meta_path = list(Path(data_path).rglob('*.json'))[0]

        with open(meta_path , "r") as file: 
            metadata = json.load(file)
        #-------------------------


        # Creating object from metadata
        #-------------------------
        for key in metadata.keys():
            setattr(self, key, metadata[key])

            # Exceptions for specific objects
            if key == 'W_matrix':
                self.W_matrix  = np.array(metadata['W_matrix'])
            elif key == 'particle_on_co':
                self.particle_on_co = xp.Particles.from_dict(metadata['particle_on_co'])
        #-------------------------

        # Importing main dataframe
        #-------------------------
        if self.parquet_data == '_df':
            self._df = import_parquet(data_path,partition_name=partition_name,partition_ID=partition_ID,variables = variables,start_at_turn=start_at_turn,stop_at_turn=stop_at_turn,handpick_particles = handpick_particles)
            self._df = coordinate_table(self._df,W_matrix=self.W_matrix,particle_on_co=self.particle_on_co,nemit_x=self.nemitt_x,nemit_y=self.nemitt_y,nemit_zeta=self.nemitt_zeta)
            self.start_at_turn = self.df.turn.min()
            self.stop_at_turn  = self.df.turn.max()
            self.n_turns       = self.stop_at_turn - self.start_at_turn
        elif (self.parquet_data == '_data') or (self.parquet_data == '_calculations'):
            self._data = import_parquet(data_path,partition_name=partition_name,partition_ID=partition_ID,variables = variables,start_at_turn=start_at_turn,stop_at_turn=stop_at_turn,handpick_particles = handpick_particles)
            self.start_at_turn = self._data.start_at_turn.min()
            self.stop_at_turn  = self._data.stop_at_turn.max()
            self.n_turns       = self.stop_at_turn - self.start_at_turn

        elif (self.parquet_data == '_checkpoint'):
            self._checkpoint = import_parquet(data_path,partition_name=partition_name,partition_ID=partition_ID,variables = variables,start_at_turn=start_at_turn,stop_at_turn=stop_at_turn,handpick_particles = handpick_particles)
            self._checkpoint = coordinate_table(self._checkpoint,W_matrix=self.W_matrix,particle_on_co=self.particle_on_co,nemit_x=self.nemitt_x,nemit_y=self.nemitt_y,nemit_zeta=self.nemitt_zeta)
            self.start_at_turn = self.checkpoint.turn.min()
            self.stop_at_turn  = self.checkpoint.turn.max()
            self.n_turns       = self.stop_at_turn - self.start_at_turn
        #-------------------------

        return self


        
    
    def to_pickle(self,filename):
        pass
        # self.context      = None
        # self.progress     = None
        # self.monitor      = None
        # self.progress     = None
        # self.PBar     = None
        # # self._plive       = None
        # # self._pstatus     = None
        # self.run_tracking  = None

        # self._tunes    = None
        # self._tunes_n  = None

        # self._df       = None
        # self._df_n     = None
        # self._df_sig   = None

        # self._coord    = None
        # self._coord_n  = None
        # self._coord_sig= None

        # with open(filename, 'wb') as f:
        #     pickle.dump(self, f)


        
    def to_parquet(self,filename,partition_name = None,partition_ID = None,parquet_data = None,handpick_particles = None):
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
            self.df.insert(0,self.partition_name,self.partition_ID)
            if handpick_particles is not None:
                self.df[self.df.particle.isin(handpick_particles)].to_parquet(filename,    partition_cols         = [self.partition_name],
                                                                                            existing_data_behavior = 'delete_matching',
                                                                                            basename_template      = 'tracking_data_{i}.parquet')
            else:
                self.df.to_parquet(filename,    partition_cols         = [self.partition_name],
                                                existing_data_behavior = 'delete_matching',
                                                basename_template      = 'tracking_data_{i}.parquet')
            self.df.drop(columns=[self.partition_name],inplace=True)
        elif self.parquet_data == '_data':
            _ = self.data
            self.data.insert(0,self.partition_name,self.partition_ID)
            self.data.to_parquet(filename, partition_cols         = [self.partition_name],
                                                    existing_data_behavior = 'delete_matching',
                                                    basename_template      = 'processed_data_{i}.parquet')
            self.data.drop(columns=[self.partition_name],inplace=True)
        elif self.parquet_data == '_checkpoint':
            _ = self.checkpoint
            self.checkpoint.insert(0,self.partition_name,self.partition_ID)
            self.checkpoint.to_parquet(filename, partition_cols         = [self.partition_name],
                                                    existing_data_behavior = 'delete_matching',
                                                    basename_template      = 'checkpoint_{i}.parquet')
            self.checkpoint.drop(columns=[self.partition_name],inplace=True)
        #---------------------------------------

        # Export metadata as well
        metadata = self.to_dict()
        
        meta_path = f'{filename}/{self.partition_name}={self.partition_ID}/meta_data.json'
        with open(meta_path , "w") as outfile: 
            json.dump(metadata, outfile,cls=NpEncoder)

    @property
    def sig_x(self):
        if self.W_matrix is not None:         
            return self.W_matrix[0,0]*np.sqrt(self.nemitt_x/self.particle_on_co.gamma0[0])
        return None
    
    @property
    def betx(self):
        if self.W_matrix is not None:         
            return (self.W_matrix[0,0])**2
        return None

    @property
    def sig_y(self):
        if self.W_matrix is not None:         
            return self.W_matrix[2,2]*np.sqrt(self.nemitt_y/self.particle_on_co.gamma0[0])
        return None
    
    @property
    def bety(self):
        if self.W_matrix is not None:         
            return (self.W_matrix[2,2])**2
        return None

    @property
    def sig_x_coll(self):
        _sigx = np.sqrt(self.betx*3.5e-6/self.particle_on_co.gamma0[0])
        return _sigx
    
    @property
    def sig_y_coll(self):
        _sigy = np.sqrt(self.bety*3.5e-6/self.particle_on_co.gamma0[0])
        return _sigy
    
    @property
    def sig_skew_coll(self):
        # Ellipse in polar: r(alpha) = sqrt((a*cos(alpha))^2 + (b*sin(alpha))^2)
        _sigskew = np.sqrt((self.sig_x_coll*np.cos(self.coll_alpha))**2 + (self.sig_y_coll*np.sin(self.coll_alpha))**2)
        return _sigskew
    
    @property
    def coll_alpha(self):
        return np.deg2rad(127.5)

    @property
    def coord(self):
        keep_col = ['particle','state','x','px','y','py','zeta','pzeta']
        if self._coord is None:
            if self._checkpoint is not None:
                self._coord = self.checkpoint.groupby('turn').get_group(0).reset_index(drop=True)
            else:
                self._coord = self.df.groupby('turn').get_group(0).reset_index(drop=True)
            self._coord = self._coord[keep_col]
        if type(self._coord) is not coordinate_table:
            self._coord = coordinate_table(self._coord,W_matrix=self.W_matrix,particle_on_co=self.particle_on_co,nemit_x=self.nemitt_x,nemit_y=self.nemitt_y,nemit_zeta=self.nemitt_zeta)
        return self._coord.df
    
    @property
    def coord_n(self):
        if type(self._coord) is not coordinate_table:
            _ = self.coord
        return self._coord.df_n
    
    @property
    def coord_sig(self):
        if type(self._coord) is not coordinate_table:
            _ = self.coord
        return self._coord.df_sig

    @property
    def data(self):
        return self._data
    
    @property
    def checkpoint(self):
        if type(self._checkpoint) is not coordinate_table:
            self._checkpoint = coordinate_table(self._checkpoint,W_matrix=self.W_matrix,particle_on_co=self.particle_on_co,nemit_x=self.nemitt_x,nemit_y=self.nemitt_y,nemit_zeta=self.nemitt_zeta)
        return self._checkpoint.df
    
    @property
    def checkpoint_n(self):
        if type(self._checkpoint) is not coordinate_table:
            _ = self.checkpoint
        return self._checkpoint.df_n
    
    @property
    def checkpoint_sig(self):
        if type(self._checkpoint) is not coordinate_table:
            _ = self.checkpoint
        return self._checkpoint.df_sig



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
            self._df = coordinate_table(self._df,W_matrix=self.W_matrix,particle_on_co=self.particle_on_co,nemit_x=self.nemitt_x,nemit_y=self.nemitt_y,nemit_zeta=self.nemitt_zeta)
        return self._df.df

    @property
    def df_n(self):
        return self._df.df_n
    

    @property
    def df_sig(self):
        return self._df.df_sig
    
    @property
    def tunes(self):
        # Reset if method is changed
        if self._tunesMTD != self._oldTunesMTD:
            self._tunes   = None
            self._tunes_n = None

        if self._tunes is None:
            if self._tunesMTD == 'nafflib':
                self._oldTunesMTD = 'nafflib'
                self._tunes    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':nafflib.tune(_part['x'], _part['px'], window_order=2, window_type="hann"),
                                                                                            'Qy':nafflib.tune(_part['y'], _part['py'], window_order=2, window_type="hann")}))
        
        return self._tunes

    @property
    def tunes_n(self):
        # Reset if method is changed
        if self._tunesMTD != self._oldTunesMTD:
            self._tunes   = None
            self._tunes_n = None

        if self._tunes_n is None:
            if self._tunesMTD == 'nafflib':
                self._oldTunesMTD = 'nafflib'
                self._tunes_n    = self.df_n.groupby('particle').apply(lambda _part: pd.Series({'Qx':nafflib.tune(_part['x_n'], _part['px_n'], window_order=2, window_type="hann"),
                                                                                                'Qy':nafflib.tune(_part['y_n'], _part['py_n'], window_order=2, window_type="hann")}))
        
        return self._tunes_n

    def compute_intensity(self,coll_opening=5,from_df='_data',at_turn = None,find_plane = False):
        # Collimator opening
        coll_x = coll_opening*self.sig_x_coll
        coll_y = coll_opening*self.sig_y_coll
        coll_s = coll_opening*self.sig_skew_coll


        def lost_condition(x_min,y_min,skew_min,x_max,y_max,skew_max):
            return ((np.abs(x_min)>coll_x)|(np.abs(y_min)>coll_y)|(np.abs(skew_min)>coll_s)|
                    (np.abs(x_max)>coll_x)|(np.abs(y_max)>coll_y) |(np.abs(skew_max)>coll_s))

        # def plane_lost(df):
        #     _plane  = pd.Series('',index=df.x_min.index)
        #     idx_x   = _plane.index[(np.abs(df.x_min)>coll_x)|(np.abs(df.x_max)>coll_x)]
        #     idx_y   = _plane.index[(np.abs(df.y_min)>coll_y)|(np.abs(df.y_max)>coll_y)]
        #     idx_skew= _plane.index[(np.abs(df.skew_min)>coll_s)|(np.abs(df.skew_max)>coll_s)]

        #     _plane.loc[idx_x] += 'x'
        #     _plane.loc[idx_y] += 'y'
        #     _plane.loc[idx_skew] += 's'

        #     return _plane
        
        # Keep columns
        coordinates = ['x','y','skew']
        keep_cols   = [f'{i}_min' for i in coordinates] + [f'{i}_max' for i in coordinates]
        keep_cols   = ['Chunk ID','particle','start_at_turn','stop_at_turn'] + keep_cols
        
        if from_df == '_data':
            group  = self.data[keep_cols]
            if at_turn is not None:
                group  = group[group.start_at_turn <= at_turn]
        elif from_df == '_checkpoint':
            #TODO
            pass
        elif from_df == '_df':
            #TODO
            pass

        _lost        = lost_condition(group.x_min,group.y_min,group.skew_min,group.x_max,group.y_max,group.skew_max)
        # _plane_lost  = plane_lost(group)
        # _lost        = _plane_lost.apply(lambda plane_str: len(plane_str)>0)
        idx_lost     = group.index[_lost]
        idx_survived = group.index[~_lost]

        # New columns
        group.insert(0,'beyond_coll',False)
        group.insert(0,'lost',False)

        group.loc[idx_lost,'beyond_coll'] = True
        group.loc[:,'lost'] = group.groupby('particle').beyond_coll.cumsum().astype(bool)

        # # Finding lost plane:
        # if find_plane:
        #     group.insert(0,'plane',_plane_lost)
        #     group.loc[_lost,'plane'] += '|'
        #     _plane_df = group[['particle','plane']]
        #     _plane_result = _plane_df.groupby('particle')['plane'].apply(pd.Series.cumsum).apply(lambda _str: _str.split('|')[0]).to_frame()
        #     _plane_result.insert(0,'index',_plane_result.index.get_level_values(1))
        #     _plane_result = _plane_result.sort_values('index').set_index('index')
        #     group.loc[:,'plane'] = _plane_result['plane']

        intensity = group[~group.lost].groupby('start_at_turn').count().particle
        intensity = group[~group.lost].groupby('start_at_turn').count().particle.to_frame()
        intensity.insert(0,'stop_at_turn',group.groupby('start_at_turn').stop_at_turn.max())
        intensity.insert(1,'Chunk ID',group.groupby('start_at_turn')['Chunk ID'].max())
        intensity.reset_index(drop=False,inplace=True)
        intensity.rename(columns={'particle':'count'},inplace=True)
        

        # Adding survived list
        survived = pd.Series(len(intensity.index)*[[]],index=intensity.index)
        _tmp     = group[~group.lost].groupby('start_at_turn').apply(lambda group: list(group.particle.values))
        if type(_tmp) is pd.Series:
            survived.loc[:] = _tmp.values
        intensity.insert(3,'survived',survived.values)

        # Appending to intensity
        starting_point = pd.DataFrame({'Chunk ID':[-1],'start_at_turn':[-1],'stop_at_turn':[0],'count':[len(group.particle.unique())],'survived':[list(group.particle.unique())]})
        intensity      = pd.concat([starting_point,intensity]).reset_index(drop=True)
        return intensity

    def initialize_monitor(self,start_at_turn=0,nturns = 1):
        monitor = xt.ParticlesMonitor( _context = self.context,num_particles = self.n_parts,
                                                start_at_turn    = start_at_turn, 
                                                stop_at_turn     = start_at_turn + nturns)
        return monitor

    def run_tracking(self,_line,particles):

        # Initiating monitor
        #-------------------------------
        if self.monitor is None:
            last_turn    = self.context.nparray_from_context_array(particles.at_turn).max()
            self.monitor = self.initialize_monitor(start_at_turn=last_turn,nturns = self.n_turns)
        #-------------------------------

        # Saving turn infos
        self.start_at_turn = self.monitor.start_at_turn
        self.stop_at_turn  = self.monitor.stop_at_turn


        if not self.progress:
            # Regular tracking if no progress needed
            _line.track(particles, num_turns=self.n_turns,turn_by_turn_monitor=self.monitor)

        else:
            
            # Splitting in desired progress chunk, or turn-by-turn
            # Note: there is always 1 single turn to start with to get a time estimate
            #-------------------------
            if self.progress_divide is not None:
                chunks = [1] + split_in_chunks(self.n_turns-1,n_chunks = self.progress_divide)
            else:
                chunks = [1] + split_in_chunks(self.n_turns-1,main_chunk=1)
            #--------------------------

            
            # PBAR
            #-------------------
            if not self.PBar.main_task.started:
                self.PBar.start()
            #-------------------

            # TRACKING
            for chunk in chunks:
                if chunk == 0:
                    continue

                # Regular tracking with num_turns = chunk
                #---------------
                _line.track(particles, num_turns=chunk,turn_by_turn_monitor=self.monitor)
                _ = self.monitor.stop_at_turn # Dummy access to data for time clock
                #---------------                

                self.PBar.update(chunk=chunk)


            #-------------------------
            if self.PBar.main_task.finished:
                self.PBar.close()
            #-------------------------

            # Saving the last task as exec time (either subtask or main task)
            self.exec_time = self.PBar.Progress.tasks[-1].finished_time


    def __repr__(self,):
        rich.inspect(RenderingTracker(self),title='Tracking_Interface', docs=False)
        return ''
#===================================================







class RenderingTracker():   
    def __init__(self,trck):
        _dct = trck.to_dict()
        skip = ['config','W_matrix']
        for key in _dct.keys():
            if key in skip:
                continue
            setattr(self, key, _dct[key])

        self.particle_on_co = str(type(trck.particle_on_co))
