
import json
import rich
import re
import numpy as np
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn,TimeElapsedColumn,SpinnerColumn,TimeRemainingColumn
import pickle
import dask.dataframe as dd
import gc
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



# NEW Tracking class:
#===================================================
class Tracking_Interface():
    
    def __init__(self,line=None,particles=None,n_turns=None,method='6D',progress=False,_context=None,
                            monitor=None,rebuild = False,extract_columns = None,skip_extraction = False,
                            nemitt_x = None,nemitt_y = None,nemitt_zeta = None,partition_name = None,partition_ID = None):
        
        # Tracking
        #-------------------------
        self.partition_name = partition_name 
        self.partition_ID   = partition_ID 

        if n_turns is not None:
            self.n_turns   = int(n_turns)
        else:
            self.n_turns   = None
        if particles is not None:
            self.n_parts   = len(particles.particle_id)
        else:
            self.n_parts   = None

        self.df        = None
        self._df_n     = None
        self._df_sig   = None

        self._coord    = None
        self._coord_n  = None
        self._coord_sig= None

        self.context         = _context
        self.skip_extraction = skip_extraction

        if extract_columns is None:
            self.extract_columns = ['at_turn','particle_id','x','px','y','py','zeta','pzeta','state','at_element']
        

        # Saving emittance
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.nemitt_y = nemitt_zeta
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
        self.progress  = progress
        self._plive    = None
        self._pstatus  = None
        #-------------------------


        # Create monitor if needed
        #--------------------------
        if (monitor is None) and (particles is not None):
            self.monitor = xt.ParticlesMonitor( _context         = self.context,
                                                start_at_turn    = 0, 
                                                stop_at_turn     = self.n_turns,
                                                n_repetitions    = 1,
                                                repetition_period= 1,
                                                num_particles    = len(particles.particle_id))
        else:
            self.monitor = monitor
        #-------------------------


        # Rebuilt tracker and attach to context
        #-------------------------
        if line is not None:
            if rebuild:
                line.discard_tracker()
                line.build_tracker(_context=self.context)
                particles.move(_context=self.context)
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
        self.method = method.lower()
        if line is not None:
            assert (method.lower() in ['4d','6d']), 'method should either be 4D or 6D (default)'
            try:
                self.runTracking(line,particles,method=method.lower())
            except Exception as error:
                self.closeLiveDisplay()
                print("An error occurred:", type(error).__name__, "–", error)
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
        self.df = import_parquet(data_path,partition_name=partition_name,partition_ID=partition_ID,variables = variables)

        meta_path = f'{data_path}/{partition_name}={partition_ID}/meta_data.json'

        with open(meta_path , "r") as file: 
            metadata = json.load(file)

        self.partition_name  = metadata['partition_name']
        self.partition_ID    = metadata['partition_ID']
        self.n_turns         = metadata['n_turns']
        self.n_parts         = metadata['n_parts']
        self.nemitt_x        = metadata['nemitt_x']
        self.nemitt_y        = metadata['nemitt_y']
        self.nemitt_zeta     = metadata['nemitt_zeta']
        self.method          = metadata['method']

        self.W_matrix        = np.array(metadata['W_matrix'])
        self.particle_on_co  = xp.Particles.from_dict(metadata['particle_on_co'])

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

        self._df_n     = None
        self._df_sig   = None

        self._coord    = None
        self._coord_n  = None
        self._coord_sig= None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def to_parquet(self,filename,partition_name = None,partition_ID = None):
        if partition_name is not None:
            self.partition_name = partition_name  
        if partition_ID is not None:
            self.partition_ID   = partition_ID

        # Export to parquet, partitioned in sub folder
        #---------------------------------------
        self.df.insert(0,self.partition_name,self.partition_ID)
        self.df.to_parquet(filename,    partition_cols         = [self.partition_name],
                                        existing_data_behavior = 'delete_matching',
                                        basename_template      = 'tracking_data_{i}.parquet')
        self.df.drop(columns=[self.partition_name],inplace=True)
        #---------------------------------------

        # Export metadata as well
        metadata = {'partition_name'  : self.partition_name,
                    'partition_ID'    : self.partition_ID,
                    'n_turns'         : self.n_turns,
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


    def runTracking(self,line,particles,method = '6d'):

        if method=='4d':
            line.freeze_longitudinal(True)
            
        if self.progress:
            
            # Run turn-by-turn to show progress
            self.startProgressBar()
            #-------------------------
            for iturn in range(self.n_turns):
                line.track(particles,turn_by_turn_monitor=self.monitor)
                self.updateProgressBar()
            #-------------------------
            self.closeLiveDisplay()


        else:
            self.startSpinner()

            line.track(particles, num_turns=self.n_turns,turn_by_turn_monitor=self.monitor)
            
            self.updateLiveDisplay()
            self.closeLiveDisplay()


        if not self.skip_extraction:

            #CONVERT TO PANDAS
            self.df = pd.DataFrame(self.monitor.to_dict()['data'])
            
            # Getting rid of lost particles
            self.df = self.df[self.df['state'] != 0].reset_index(drop=True)

            # Filter the data
            self.df.insert(list(self.df.columns).index('zeta'),'pzeta',self.df['ptau']/self.df['beta0'])
            self.df = self.df[self.extract_columns]
            self.df.rename(columns={"at_turn": "turn",'particle_id':'particle'},inplace=True)

            # Adding element name
            if 'at_element' in self.extract_columns:
                self.df.loc[:,'at_element'] = self.df.at_element.apply(lambda ee_idx: line.element_names[ee_idx])
        


        # Unfreeze longitudinal
        if method=='4d':
            line.freeze_longitudinal(False)



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
    
    def updateProgressBar(self,):
        self._plive.update(self._pstatus, advance=1,update=True)

    def updateLiveDisplay(self,):
        self._plive.update(self._pstatus,advance=1,update=True)
        
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

#===================================================








# OLD Tracking class:
#===================================================
class Tracking():
    
    def __init__(self,tracker,particles,n_turns,method='6D',progress=False,saveVars = False):
        
        #self.particles = particles.copy()
        self.n_turns   = int(n_turns)
        self.vars      = None

        # Footprint info
        self._tunes    = None
        self._tunes_n  = None
        self._tunesMTD    = 'pynaff'
        self._oldTunesMTD = 'pynaff'

        # Savevars if needed
        if saveVars:
            self.vars = tracker.vars.copy()

        # Progress info
        self.progress  = progress
        self._plive    = None
        self._pstatus  = None
        
        # Tracking
        self.monitor   = None
        self.df        = None

        assert (method.lower() in ['4d','6d']), 'method should either be 4D or 6D (default)'
        try:
            self.runTracking(tracker,particles,method=method.lower())
        except KeyboardInterrupt:
            self.closeLiveDisplay()


        # Disabling Tracking
        self.runTracking = lambda _: print('New Tracking instance needed')
    
    def to_pickle(self,filename):
        self.progress     = None
        self.monitor      = None
        self.progress     = None
        self._plive       = None
        self._pstatus     = None
        self.runTracking  = None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

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


    def runTracking(self,tracker,particles,method = '6d'):

        if method=='4d':
            config = xt.tracker.TrackerConfig()
            config.update(tracker.config)

            _tracker = tracker
            _tracker.freeze_longitudinal(True)
            
            # Some checks
            assert _tracker.line is tracker.line
            assert _tracker._buffer is tracker._buffer
            
            
        else:
            _tracker = tracker

        if self.progress:
            # Create monitor if needed
            if self.monitor is None:
                self.monitor = xt.ParticlesMonitor( start_at_turn    = 0, 
                                                    stop_at_turn     = self.n_turns,
                                                    n_repetitions    = 1,
                                                    repetition_period= 1,
                                                    num_particles    = len(particles.particle_id))

            # Run turn-by-turn to show progress
            self.startProgressBar()
            #-------------------------
            for iturn in range(self.n_turns):
                self.monitor.track(particles)
                _tracker.track(particles)
                self.updateProgressBar()
            #-------------------------
            self.closeLiveDisplay()

            #CONVERT TO PANDAS
            self.df = pd.DataFrame(self.monitor.to_dict()['data'])

        else:
            # self.startSpinner()
            _tracker.track(particles, num_turns=self.n_turns,turn_by_turn_monitor=True)
            # self.closeLiveDisplay()

            #CONVERT TO PANDAS
            self.df = pd.DataFrame(_tracker.record_last_track.to_dict()['data'])
        
        # Filter the data
        self.df.insert(list(self.df.columns).index('zeta'),'pzeta',self.df['ptau']/self.df['beta0'])
        self.df = self.df[['at_turn','particle_id','x','px','y','py','zeta','pzeta','state']]
        self.df.rename(columns={"at_turn": "turn",'particle_id':'particle'},inplace=True)

        # Return in normalized space as well:
        # NOTE: twiss can only be done on tracker6D!!
        coord_n = W_phys2norm(**self.df[['x','px','y','py','zeta','pzeta']],twiss=_tracker.twiss(method=method),to_pd=True)
        self.df = pd.concat([self.df,coord_n],axis=1)

        # Unfreeze longitudinal
        if method=='4d':
            tracker.config = config

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
    
    def updateProgressBar(self,):
        self._plive.update(self._pstatus, advance=1,update=True)

        
    def startSpinner(self,):
        self._plive = Progress("{task.description}",
                                SpinnerColumn(),
                                TextColumn("[progress.elapsed] ["),TimeElapsedColumn (),TextColumn("[progress.elapsed]elapsed ]   "))

        self._plive.start()
        self._plive.live._disable_redirect_io()

        self._pstatus = self._plive.add_task("[blue]Tracking", total=self.n_turns)


    def closeLiveDisplay(self,):
        self._plive.refresh()
        self._plive.stop()
        self._plive.console.clear_live()

#===================================================