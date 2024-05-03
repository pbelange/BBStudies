
import json
import rich
import numpy as np
import pandas as pd
import traceback
from pathlib import Path
import nafflib

import xtrack as xt
import xpart as xp
import xobjects as xo



import BBStudies.Tracking.Progress as PBar
import BBStudies.Tracking.Utils as xutils
import BBStudies.Physics.Constants as cst



class Poincare_Section():
    def __init__(self,name=None,twiss = None,tune_on_co=None,nemitt_x=None,nemitt_y=None,nemitt_zeta=None):
        #===========================
        self.name           = name
        self.data           = {'naff':None,'excursion':None,'checkpoints':None,'tbt':None}
        #-------------
        self.twiss          = twiss
        self.tune_on_co     = tune_on_co
        self.nemitt_x       = nemitt_x
        self.nemitt_y       = nemitt_y
        self.nemitt_zeta    = nemitt_zeta
        #===========================


    
    @property
    def datakeys(self):
        return [key for key in self.data.keys() if self.data[key] is not None]

    def to_dict(self):
        metadata = {'name'            : self.name,
                    'datakeys'        : self.datakeys,
                    'twiss'           : self.twiss.to_dict(),
                    'nemitt_x'        : self.nemitt_x,
                    'nemitt_y'        : self.nemitt_y,
                    'nemitt_zeta'     : self.nemitt_zeta,
                    'tune_on_co'      : self.tune_on_co}
        return metadata


    def to_parquet(self,path,collider_name,distribution_name,datakeys = None):

        if datakeys is None:
            datakeys = self.data_keys

        for key in datakeys:

            self.data[key].insert(0,'collider'      ,collider_name)
            self.data[key].insert(0,'distribution'  ,distribution_name)
            self.data[key].insert(0,'at_element'    ,self.name)
            self.data[key].insert(0,'data'          ,key)

            chunk_label = 'window' if key=='naff' else 'chunk'

            self.data[key].to_parquet(path, partition_cols          = ['collider','distribution','at_element','data',chunk_label],
                                            existing_data_behavior  = 'delete_matching',
                                            basename_template       = key + '_datafile_{i}.parquet')


        # Export metadata for the object
        metadata = self.to_dict()
        meta_path = f'{path}/collider={collider_name}/distribution={distribution_name}/at_element={self.name}/poincare_metadata.json'
        xutils.mkdir(Path(meta_path).parent)
        with open(meta_path , "w") as outfile: 
            json.dump(metadata, outfile,cls=xutils.NpEncoder)

    @classmethod
    def from_parquet(cls,path,collider_name,distribution_name,name,datakeys=None):
        self = cls()
        
        
        # Creating object from metadata
        #-------------------------
        meta_path = f'{path}/collider={collider_name}/distribution={distribution_name}/at_element={name}/poincare_metadata.json'

        with open(meta_path , "r") as file: 
            metadata = json.load(file)

        for key in metadata.keys():
            # Exceptions for specific objects
            if key == 'twiss':
                self.twiss  = xt.TwissInit.from_dict(metadata['twiss']) 
            elif key == 'datakeys':
                meta_datakeys = metadata['datakeys']
            else:
                setattr(self, key, metadata[key])
        #-------------------------

        # Loading datakeys
        #-------------------------
        partition_dict = {  'collider'    : collider_name,
                            'distribution': distribution_name,
                            'at_element'  : name,
                            'data'        : None}

        if datakeys is not None:
            meta_datakeys = datakeys


        for key in meta_datakeys:
            if key == 'naff':
                complex_columns = ['Ax','Ay','Azeta']
            else:
                complex_columns = None
            
            # Loading data from parquet
            # partition_dict['data'] = key
            # _df = xutils.import_parquet_datafile(path,partition_dict = partition_dict,complex_columns=complex_columns)
            partition_dict = {'data':key}
            _df = xutils.import_parquet_datafile(f'{path}/collider={collider_name}/distribution={distribution_name}/at_element={name}',
                                                    partition_dict = partition_dict,
                                                    complex_columns=complex_columns)

            # Converting to coordinate table if needed
            if key in ['checkpoints','tbt']:
                _df = coordinate_table(_df,twiss=self.twiss,nemitt_x=self.nemitt_x,nemitt_y=self.nemitt_y,nemitt_zeta=self.nemitt_zeta)
            
            # saving in data container
            self.data[key] = _df
        #-------------------------
            
        return self

    @property
    def ee_name(self):
        return self.twiss.element_name
    
    @property
    def s(self):
        return self.twiss.s
    
    @property
    def W_matrix(self):
        return self.twiss.W_matrix
    
    @property
    def particle_on_co(self):
        return self.twiss.particle_on_co
    


    
    def __repr__(self,):
        print(' ')
        rich.inspect(RenderingPoincare(self),title='Poincare_Section', docs=False)
        return ''
    
class RenderingPoincare():   
    def __init__(self,poincare):
        _dct = poincare.to_dict()
        skip = ['twiss']
        for key in _dct.keys():
            if key in skip:
                continue
            setattr(self, key, _dct[key])
        # self.particle_on_co = poincare.W_matrix.tolist()
        # self.particle_on_co = str(type(poincare.particle_on_co))
        
    



class Tracking_Interface():
    
    def __init__(self,  line            = None,
                        method          = '6D',
                        cycle_at        = None,
                        sequence        = None,
                        context         = None,
                        config          = None,

                        num_particles   = None,
                        num_turns       = None,

                        nemitt_x        = None,
                        nemitt_y        = None,
                        nemitt_zeta     = None,
                        sigma_z         = None,

                        poincare        = [],
                        
                        PBar            = None,
                        progress        = False,
                        progress_divide = 100,):

        # Saving attributes:
        #-------------------------
        self.method         = method
        self.cycle_at       = cycle_at
        self.sequence       = sequence
        self.context        = context
        self.config         = config

        self.num_particles  = num_particles
        self.num_turns      = num_turns

        self.nemitt_x       = nemitt_x
        self.nemitt_y       = nemitt_y
        self.nemitt_zeta    = nemitt_zeta
        self.sigma_z        = sigma_z
        
        self.poincare       = poincare

        self.progress_divide = progress_divide
        self.progress       = progress
        self.PBar           = PBar
        #-------------------------

        # Custom attributes:
        #-------------------------
        self.context_name   = self.context.__class__.__name__

        if self.num_turns is not None:
            self.num_turns   = int(self.num_turns)
        if self.num_particles is not None:
            self.num_particles = int(self.num_particles)
        #-------------------------
            

        # Progress info
        #-------------------------
        if (PBar is None) and (progress):
            self.PBar = PBar.ProgressBar(message = 'Tracking ...',color='blue',n_steps = self.n_turns)
        elif PBar is not None:
            self.progress = True
        else:
            self.PBar     = None
            self.progress = False
        self.exec_time = None
        #-------------------------


        # Relevant twiss information
        #--------------------------
        if line is not None:
            _twiss = line.twiss(method=method.lower())
            self.twiss = _twiss.get_twiss_init(at_element=line.element_names[0])
            self.tune_on_co     = [_twiss.mux[-1], _twiss.muy[-1], _twiss.muzeta[-1]]
        else:
            self.twiss          = None
            self.tune_on_co     = None
        #--------------------------


    def to_dict(self,):
        metadata = {'sequence'        : self.sequence,
                    'cycle_at'        : self.cycle_at,
                    'method'          : self.method,
                    'context_name'    : self.context_name,
                    'exec_time'       : self.exec_time,
                    'num_particles'   : self.num_particles,
                    'num_turns'       : self.num_turns,
                    'nemitt_x'        : self.nemitt_x,
                    'nemitt_y'        : self.nemitt_y,
                    'nemitt_zeta'     : self.nemitt_zeta,
                    'sigma_z'         : self.sigma_z,
                    'poincare'        : {poincare.name:poincare.datakeys for poincare in self.poincare},
                    'twiss'           : self.twiss.to_dict(),
                    'tune_on_co'      : self.tune_on_co,
                    'config'          : self.config}
        return metadata

    def export_metadata(self,path,collider_name,distribution_name):
        # Export metadata for the object
        metadata = self.to_dict()
        meta_path = f'{path}/collider={collider_name}/distribution={distribution_name}/interface_metadata.json'

        xutils.mkdir(Path(meta_path).parent)

        with open(meta_path , "w") as outfile: 
            json.dump(metadata, outfile,cls=xutils.NpEncoder)

    @classmethod
    def from_parquet(cls,path,collider_name,distribution_name,poincare_names = None,datakeys = None):
        self = cls()
        
        
        # Creating object from metadata
        #-------------------------
        meta_path = f'{path}/collider={collider_name}/distribution={distribution_name}/interface_metadata.json'

        with open(meta_path , "r") as file: 
            metadata = json.load(file)

        for key in metadata.keys():
            # Exceptions for specific objects
            if key == 'twiss':
                self.twiss  = xt.TwissInit.from_dict(metadata['twiss']) 
            elif key == 'poincare':
                meta_poincare = list(metadata['poincare'].keys())
            else:
                setattr(self, key, metadata[key])
        #-------------------------


        # Loading poincare
        #-------------------------
        if poincare_names is not None:
            meta_poincare = poincare_names
        
        self.poincare = []
        for _name in meta_poincare:
            self.poincare.append(Poincare_Section.from_parquet( path                = path,
                                                                collider_name       = collider_name,
                                                                distribution_name   = distribution_name,
                                                                name                = _name,
                                                                datakeys            = datakeys))
        
        return self



    def run_tracking(self,line,particles,num_turns = 1,method = None):
        if method is not None:
            self.method = method
        assert (self.method.lower() in ['4d','6d']), 'method should either be 4D or 6D (default)'
        try:
            if self.method.lower()=='4d':
                line.freeze_longitudinal(True)

            # Track
            #=================
            self._tracking(line,particles,num_turns = num_turns)
            #=================

            # Unfreeze longitudinal
            if self.method.lower()=='4d':
                line.freeze_longitudinal(False)

        except Exception as error:
            self.PBar.close()
            print("An error occurred:", type(error).__name__, " - ", error)
            traceback.print_exc()
        except KeyboardInterrupt:
            self.PBar.close()
            print("Terminated by user: KeyboardInterrupt")



    def _tracking(self,line,particles,num_turns = 1):


        if not self.progress:
            # Regular tracking if no progress needed
            line.track(particles, num_turns=num_turns)

        else:
            # Splitting in desired progress chunk, or turn-by-turn
            # Note: there is always 1 single turn to start with to get a time estimate
            #-------------------------
            if self.progress_divide is not None:
                chunks = [1] + split_in_chunks(num_turns-1,n_chunks = self.progress_divide)
            else:
                chunks = [1] + split_in_chunks(num_turns-1,main_chunk=1)
            #--------------------------

            
            # PBar
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
                line.track(particles, num_turns=chunk)
                _ = particles.at_turn # Dummy access to data for time clock
                #---------------                

                self.PBar.update(chunk=chunk)


            #-------------------------
            if self.PBar.main_task.finished:
                self.PBar.close()
                self.exec_time = self.PBar.main_task.finished_time
            else:
                self.exec_time = self.PBar.Progress.tasks[-1].finished_time
            #-------------------------

            # Saving the last task as exec time (either subtask or main task)
            

    @property
    def prettyconfig(self):
        rich.inspect(self.config,title='Config',docs=False)
        return ''
    
    @property
    def W_matrix(self):
        return self.twiss.W_matrix
    
    @property
    def particle_on_co(self):
        return self.twiss.particle_on_co

    def __repr__(self,):
        print(' ')
        rich.inspect(RenderingInterface(self),title='Tracking_Interface', docs=False)
        return ''
    
class RenderingInterface():   
    def __init__(self,trck):
        _dct = trck.to_dict()
        skip = ['config','twiss']
        for key in _dct.keys():
            if key in skip:
                continue
            setattr(self, key, _dct[key])

        # self.particle_on_co = str(type(trck.particle_on_co))
#===================================================




#===================================================
class coordinate_table():
    def __init__(self,df,twiss,nemitt_x=None,nemitt_y=None,nemitt_zeta=None):
        self.df      = df
        self._df_n   = None
        self._df_sig = None

        self.twiss    = twiss
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.nemitt_zeta = nemitt_zeta


    @property
    def df_n(self):
        if self._df_n is None:
            XX_n       =  _W_phys2norm(**self.df[['x','px','y','py','zeta','pzeta']],W_matrix=self.twiss.W_matrix,co_dict = self.twiss.particle_on_co.copy(_context=xo.context_default).to_dict())
            coord_n    = pd.DataFrame({'x_n':XX_n[0,:],'px_n':XX_n[1,:],'y_n':XX_n[2,:],'py_n':XX_n[3,:],'zeta_n':XX_n[4,:],'pzeta_n':XX_n[5,:]})
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
            XX_sig          =  _W_phys2norm(**self.df[['x','px','y','py','zeta','pzeta']],nemitt_x= self.nemitt_x, nemitt_y= self.nemitt_y, nemitt_zeta= self.nemitt_zeta,
                                         W_matrix=self.twiss.W_matrix,co_dict = self.twiss.particle_on_co.copy(_context=xo.context_default).to_dict())
            
            coord_sig       = pd.DataFrame({'x_sig':XX_sig[0,:],'px_sig':XX_sig[1,:],'y_sig':XX_sig[2,:],'py_sig':XX_sig[3,:],'zeta_sig':XX_sig[4,:],'pzeta_sig':XX_sig[5,:]})
            old_cols        = list(self.df.columns.drop(['x','px','y','py','zeta','pzeta']))
            self._df_sig    = pd.concat([self.df[old_cols],coord_sig],axis=1)
        return self._df_sig
#===================================================




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
        
        _twiss = line.twiss(method='6d')
        dct_longitudinal['momentum_compaction_factor'] = _twiss['momentum_compaction_factor']

        self.W_matrix = _twiss.W_matrix[0]
        self.particle_on_co = _twiss.particle_on_co.copy(_context=xo.context_default)

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

        co_dict = self.particle_on_co.to_dict()
        WW      = self.W_matrix
        betzeta = WW[4, 4]**2 + WW[4, 5]**2
        nemitt_zeta = ((sigma_z**2/betzeta) * (co_dict['beta0'] * co_dict['gamma0']))[0]

        return nemitt_zeta
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


def _W_phys2norm(x, px, y, py, zeta, pzeta, W_matrix, co_dict, nemitt_x=None, nemitt_y=None, nemitt_zeta=None):
    
    
    # Compute geometric emittances if normalized emittances are provided
    gemitt_x = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_x is None else (nemitt_x / co_dict['beta0'] / co_dict['gamma0'])
    gemitt_y = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_y is None else (nemitt_y / co_dict['beta0'] / co_dict['gamma0'])
    gemitt_zeta = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_zeta is None else (nemitt_zeta / co_dict['beta0'] / co_dict['gamma0'])

    
    # Prepaing co arrray and gemitt array:
    co = np.array([co_dict['x'], co_dict['px'], co_dict['y'], co_dict['py'], co_dict['zeta'], co_dict['ptau'] / co_dict['beta0']])
    gemitt_values = np.array([gemitt_x, gemitt_x, gemitt_y, gemitt_y, gemitt_zeta, gemitt_zeta])

    # Ensuring consistent dimensions
    for add_axis in range(-1,len(np.shape(x))-len(np.shape(co))):
        co = co[:,np.newaxis]
    for add_axis in range(-1,len(np.shape(x))-len(np.shape(gemitt_values))):
        gemitt_values = gemitt_values[:,np.newaxis]

    
    # substracting closed orbit
    XX = np.array([x, px, y, py, zeta, pzeta])
    XX -= co
    

    # Apply the inverse transformation matrix
    W_inv = np.linalg.inv(W_matrix)
    
    if len(np.shape(XX)) == 3:
        XX_norm = np.dot(W_inv, XX.reshape(6,x.shape[0]*x.shape[1]))
        XX_norm = XX_norm.reshape(6, x.shape[0], x.shape[1])
    else:    
        XX_norm = np.dot(W_inv, XX)
    
    # Normalize the coordinates with the geometric emittances
    XX_norm /= np.sqrt(gemitt_values)
    

    return XX_norm



#===================================================
def split_in_chunks(turns,main_chunk = None,n_chunks = None):

    if n_chunks is not None:
        n_chunks = int(n_chunks)

        # See https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split
        l = turns
        n = n_chunks
        chunks = (l % n) * [l//n + 1] + (n-(l % n))*[l//n]
        
    elif main_chunk is not None:
        main_chunk = int(main_chunk)

        n_chunks = turns//main_chunk
        chunks   = n_chunks*[main_chunk]+ [np.mod(turns,main_chunk)]
    
    if chunks[-1]==0:
        chunks = chunks[:-1]

    return chunks
#===================================================




