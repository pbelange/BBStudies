import numpy as np
import pandas as pd

import nafflib
import xobjects as xo
import xtrack as xt

import BBStudies.Tracking.XsuitePlus as xPlus


#===================================================
# BASE CLASS
#===================================================
class Buffer():
    def __init__(self,):
        self.call_ID = None
        # Data dict to store whatever data
        self.data = {}
        # Particle ID to keep track
        self.particle_id = None
        self.complex2tuple = True


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
                if np.issubdtype(value[0].dtype, complex) and self.complex2tuple:
                    # is complex
                    dct[key] = [[(c.real, c.imag) for c in row] for row in np.vstack(value).tolist()]
                else:
                    dct[key] = np.vstack(value).tolist()
            else:
                pass

        return dct
    
    def to_pandas(self):
        return pd.DataFrame(self.to_dict())
    

    def update(self,monitor):
        # Initialize
        #-------------------------
        if self.call_ID is None:
            self.call_ID = 0
        else:
            self.call_ID += 1
        
        if self.particle_id is None:
            self.particle_id = np.arange(monitor.part_id_start,monitor.part_id_end)
        #-------------------------
#===================================================


#===================================================
# CPU monitor
# Class to store the monitor data (only convert GPU -> CPU once)
#===================================================
class CPU_monitor():
    
    def __init__(self,):
        self.x = None
        self.px = None
        self.y = None
        self.py = None
        self.zeta = None
        self.pzeta = None
        self.state = None
        self.start_at_turn = None
        self.stop_at_turn = None
        self.part_id_start = None
        self.part_id_end = None

        
        
    def process(self,monitor):
        # Copying data: accessing the GPU monitor data copies it to CPU!
        self.x = monitor.x
        self.px = monitor.px
        self.y = monitor.y
        self.py = monitor.py
        self.zeta = monitor.zeta
        self.pzeta = np.array(np.divide(monitor.ptau,monitor.beta0, np.zeros_like(monitor.ptau) + np.nan,where=monitor.beta0!=0))#np.array(monitor.pzeta)
        self.state = monitor.state
        self.start_at_turn = monitor.start_at_turn
        self.stop_at_turn = monitor.stop_at_turn
        self.part_id_start = monitor.part_id_start
        self.part_id_end = monitor.part_id_end

        self.num_turns = len(self.x)


    # CPU monitor is always overwritten by xt.ParticleMonitor! 
    def clean(self):
        pass

    # CPU monitor is always full! 
    @property
    def is_full(self,):
        return ((self.stop_at_turn-self.start_at_turn)>=self.num_turns)
#===================================================


#===================================================
# Storage monitor
# Class to store data over several chunks
#===================================================
class storage_monitor():
    
    def __init__(self,num_particles,num_turns):
        num_particles   = int(num_particles)
        num_turns       = int(num_turns)
        
        self.num_particles  = num_particles
        self.num_turns      = num_turns

        # Initialize arrays, ordered by [particles,turns]
        self.x = np.zeros((num_particles,num_turns))
        self.px = np.zeros((num_particles,num_turns))
        self.y = np.zeros((num_particles,num_turns))
        self.py = np.zeros((num_particles,num_turns))
        self.zeta = np.zeros((num_particles,num_turns))
        self.pzeta = np.zeros((num_particles,num_turns))
        self.state = np.zeros((num_particles,num_turns))

        self.clean()


    def clean(self):

        # reset arrays
        self.x *= 0
        self.px *= 0
        self.y *= 0
        self.py *= 0
        self.zeta *= 0
        self.pzeta *= 0
        self.state *= 0

        self.start_at_turn = None
        self.stop_at_turn = None

        self.part_id_start = None
        self.part_id_end = None

        self.call_ID = None
    

    def update(self,monitor):
        # Initialize
        #-------------------------
        if self.call_ID is None:
            self.call_ID = 0
            assert monitor.part_id_end-monitor.part_id_start==self.num_particles, "num_particles is not consistent!"
            self.start_at_turn  = monitor.start_at_turn
            self.stop_at_turn   = monitor.stop_at_turn
            self.part_id_start  = monitor.part_id_start
            self.part_id_end    = monitor.part_id_end
        else:
            self.call_ID += 1
        
            assert self.part_id_start==monitor.part_id_start, "part_id_start should be the same"
            assert self.part_id_end==monitor.part_id_end, "part_id_end should be the same"
            
            # Keeping start_at_turn, updating stop_at_turn
            self.stop_at_turn = monitor.stop_at_turn

            

    def process(self,monitor):
        self.update(monitor)

        # Updating arrays
        _from = monitor.start_at_turn-self.start_at_turn
        _to   = monitor.stop_at_turn-self.start_at_turn
        self.x[:,_from:_to] = monitor.x.copy()
        self.px[:,_from:_to] = monitor.px.copy()
        self.y[:,_from:_to] = monitor.y.copy()
        self.py[:,_from:_to] = monitor.py.copy()
        self.zeta[:,_from:_to] = monitor.zeta.copy()
        self.pzeta[:,_from:_to] = monitor.pzeta.copy()
        self.state[:,_from:_to] = monitor.state.copy()

    @property
    def is_full(self,):
        return ((self.stop_at_turn-self.start_at_turn)>=self.num_turns)
#===================================================


#===================================================
# CHECKPOINT BUFFER
#===================================================
class Checkpoint_Buffer(Buffer):
    def __init__(self,):
        super().__init__()
        self.clean()
        
        
    def clean(self,):
        self.data['chunk'] = []
        self.data['particle'] = []
        self.data['turn']     = []
        self.data['state']    = []
        
        self.data['x']        = []
        self.data['px']       = []
        self.data['y']        = []
        self.data['py']       = []
        self.data['zeta']     = []
        self.data['pzeta']    = []
        

    def process(self,monitor):
        self.update(monitor = monitor)


        start_at_turn = monitor.start_at_turn

        # Appending to data
        #-------------------------
        self.data['chunk'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['turn'].append(start_at_turn)
        self.data['state'].append(monitor.state[:,-1].astype('int').copy())

        self.data['x'].append(monitor.x[:,0].copy())
        self.data['px'].append(monitor.px[:,0].copy())
        self.data['y'].append(monitor.y[:,0].copy())
        self.data['py'].append(monitor.py[:,0].copy())
        self.data['zeta'].append(monitor.zeta[:,0].copy())
        self.data['pzeta'].append(monitor.pzeta[:,0].copy())
        #-------------------------
#===================================================
        
#===================================================
# EXCURSION BUFFER
#===================================================
class Excursion_Buffer(Buffer):
    def __init__(self,):
        super().__init__()  
        self.clean()
        

    def clean(self,):        
        self.data['chunk'] = []
        self.data['particle'] = []
        self.data['start_at_turn'] = []
        self.data['stop_at_turn']  = []
        self.data['state']    = []

        self.data['x_min'] = []
        self.data['x_max'] = []
        self.data['y_min'] = []
        self.data['y_max'] = []
        self.data['zeta_min'] = []
        self.data['zeta_max'] = []
        self.data['skew_min'] = []
        self.data['skew_max'] = []


    def process(self,monitor):
        self.update(monitor = monitor)


        # Extracting data
        #-------------------------
        start_at_turn = monitor.start_at_turn
        stop_at_turn  = monitor.stop_at_turn

        x    = monitor.x
        y    = monitor.y
        zeta = monitor.zeta
         
        # Rotating for skew collimator
        #-------------------------
        skew_angle   = 127.5 + 90 #skew coll angle, +90 because 0 corresponds to a horizontal collimator_n/vertical walls
        theta_unskew = -np.deg2rad(skew_angle-90)
        x_skew       = x*np.cos(theta_unskew) - y*np.sin(theta_unskew)


        # Extracting max and min || Note: 2D array are ordered following [particles,turns]
        #-------------------------
        # X -------------
        idx_list = np.arange(len(self.particle_id))
        idx_max  = np.argmax(x,axis=1)
        idx_min  = np.argmin(x,axis=1)
        x_max = x[idx_list,idx_max]
        x_min = x[idx_list,idx_min]

        # Y -------------
        idx_max = np.argmax(y,axis=1)
        idx_min = np.argmin(y,axis=1)
        y_max = y[idx_list,idx_max]
        y_min = y[idx_list,idx_min]

        # Zeta ----------
        idx_max = np.argmax(zeta,axis=1)
        idx_min = np.argmin(zeta,axis=1)
        zeta_max = zeta[idx_list,idx_max]
        zeta_min = zeta[idx_list,idx_min]

        # Skew ----------
        skew_max = np.max(x_skew,axis=1)
        skew_min = np.min(x_skew,axis=1)
        

        # Appending to data
        #-------------------------
        self.data['chunk'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['start_at_turn'].append(start_at_turn)
        self.data['stop_at_turn'].append(stop_at_turn)
        self.data['state'].append(monitor.state[:,-1].astype('int').copy())

        self.data['x_min'].append(x_min)
        self.data['x_max'].append(x_max)
        self.data['y_min'].append(y_min)
        self.data['y_max'].append(y_max)
        self.data['zeta_min'].append(zeta_min)
        self.data['zeta_max'].append(zeta_max)
        self.data['skew_min'].append(skew_min)
        self.data['skew_max'].append(skew_max)
        #-------------------------
#===================================================
        



#===================================================
# NAFF BUFFER
#===================================================
class NAFF_Buffer(Buffer):
    def __init__(self,normalize=True,complex2tuple=True):
        super().__init__()  
        self.clean()
        self.normalize = normalize
        self.complex2tuple = complex2tuple

        # To be injected manually!
        #=========================
        self.twiss          = None
        self.nemitt_x       = None
        self.nemitt_y       = None
        self.nemitt_zeta    = None
        #=========================

        # NAFF parameters
        #=========================
        self.n_harm       = None
        self.window_order = None
        self.window_type  = None
        self.multiprocesses = None
        self.normalize = normalize
        #=========================

        
    def clean(self,):
        self.data['window'] = []
        self.data['particle'] = []
        self.data['start_at_turn'] = []
        self.data['stop_at_turn']  = []
        self.data['N'] = []
        self.data['state']    = []

        self.data['Ax']  = []
        self.data['Qx']  = []
        self.data['Ay']  = []
        self.data['Qy']  = []
        self.data['Azeta']  = []
        self.data['Qzeta']  = []

    def process(self,monitor):
        self.update(monitor = monitor)


        # Extracting data
        #-------------------------
        start_at_turn = monitor.start_at_turn
        stop_at_turn  = monitor.stop_at_turn

        x    = monitor.x
        px   = monitor.px
        y    = monitor.y
        py   = monitor.py
        zeta = monitor.zeta
        pzeta = monitor.pzeta

        if self.normalize:
            # Computing normalized coordinates
            #--------------------------
            XX_sig = xPlus._W_phys2norm(x,px,y,py,zeta,pzeta, 
                                            W_matrix    = self.twiss.W_matrix,
                                            co_dict     = self.twiss.particle_on_co.copy(_context=xo.context_default).to_dict(), 
                                            nemitt_x    = self.nemitt_x, 
                                            nemitt_y    = self.nemitt_y, 
                                            nemitt_zeta = self.nemitt_zeta)

            x_sig       = XX_sig[0,:,:]
            px_sig      = XX_sig[1,:,:]
            y_sig       = XX_sig[2,:,:]
            py_sig      = XX_sig[3,:,:]
            zeta_sig    = XX_sig[4,:,:]
            pzeta_sig   = XX_sig[5,:,:]
        else:
            x_sig       = x
            px_sig      = px
            y_sig       = y
            py_sig      = py
            zeta_sig    = zeta
            pzeta_sig   = pzeta


        # Extracting the harmonics
        #--------------------------
        n_harm       = self.n_harm
        window_order = self.window_order
        window_type  = self.window_type

        Ax,Qx       = nafflib.multiparticle_harmonics(x_sig,px_sig      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
        Ay,Qy       = nafflib.multiparticle_harmonics(y_sig,py_sig      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
        Azeta,Qzeta = nafflib.multiparticle_harmonics(zeta_sig,pzeta_sig, num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)


        # Appending to data
        #-------------------------
        self.data['window'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['start_at_turn'].append(start_at_turn)
        self.data['stop_at_turn'].append(stop_at_turn)
        self.data['N'].append(len(x_sig[0]))
        self.data['state'].append(monitor.state[:,-1].astype('int').copy())
        #----------
        self.data['Ax'].append(Ax)
        self.data['Qx'].append(Qx)
        self.data['Ay'].append(Ay)
        self.data['Qy'].append(Qy)
        self.data['Azeta'].append(Azeta)
        self.data['Qzeta'].append(Qzeta)
        #-------------------------
#===================================================



#===================================================
# TORUS BUFFER
#===================================================
class TORUS_Buffer(Buffer):
    def __init__(self,normalize=True,complex2tuple=True,skip_naff = False):
        super().__init__()  
        self.clean()
        self.normalize      = normalize
        self.complex2tuple  = complex2tuple
        self.skip_naff      = skip_naff

        # To be injected manually!
        #=========================
        self.twiss          = None
        self.nemitt_x       = None
        self.nemitt_y       = None
        self.nemitt_zeta    = None
        #=========================

        # NAFF parameters
        #=========================
        self.n_torus      = None
        self.n_points     = None
        #-------------------------
        self.n_harm       = None
        self.window_order = None
        self.window_type  = None
        self.multiprocesses = None
        #=========================

    def to_dict(self):
        dct    = {}
        for key,value in self.data.items():
            if len(value) == 0:
                continue
            if np.issubdtype(value[0].dtype, complex) and self.complex2tuple:
                # is complex
                dct[key] = [[(c.real, c.imag) for c in row] for row in value]
            else:
                dct[key] = value.tolist()
        return dct
        
    def clean(self,):
        self.data['turn']   = []
        self.data['torus']  = []
        self.data['state']  = []

        self.data['Ax']  = []
        self.data['Qx']  = []
        self.data['Ay']  = []
        self.data['Qy']  = []
        self.data['Azeta']  = []
        self.data['Qzeta']  = []

        self.data['Jx']  = []
        self.data['Jy']  = []
        self.data['Jzeta']  = []

    def process(self,monitor):
        self.update(monitor = monitor)

        assert self.call_ID <= 1, "TORUS_Buffer is not designed to store multiple chunks!"


        # Extracting data
        #-------------------------
        start_at_turn = monitor.start_at_turn
        stop_at_turn  = monitor.stop_at_turn
        self.n_turns  = stop_at_turn-start_at_turn

        x    = monitor.x
        px   = monitor.px
        y    = monitor.y
        py   = monitor.py
        zeta = monitor.zeta
        pzeta = monitor.pzeta

        if self.normalize:
            # Computing normalized coordinates
            #--------------------------
            XX_sig = xPlus._W_phys2norm(x,px,y,py,zeta,pzeta, 
                                            W_matrix    = self.twiss.W_matrix,
                                            co_dict     = self.twiss.particle_on_co.copy(_context=xo.context_default).to_dict(), 
                                            nemitt_x    = self.nemitt_x, 
                                            nemitt_y    = self.nemitt_y, 
                                            nemitt_zeta = self.nemitt_zeta)

            x_sig       = XX_sig[0,:,:]
            px_sig      = XX_sig[1,:,:]
            y_sig       = XX_sig[2,:,:]
            py_sig      = XX_sig[3,:,:]
            zeta_sig    = XX_sig[4,:,:]
            pzeta_sig   = XX_sig[5,:,:]
        else:
            x_sig       = x
            px_sig      = px
            y_sig       = y
            py_sig      = py
            zeta_sig    = zeta
            pzeta_sig   = pzeta


        # Reshaping for faster handling
        #========================================
        torus_idx,turn_idx = np.mgrid[:self.n_torus,:self.n_turns]
        torus_idx = torus_idx.reshape(self.n_torus*self.n_turns)
        turn_idx  = turn_idx.reshape(self.n_torus*self.n_turns)
        state_multi = np.all(np.array(np.split(monitor.state.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)==1,axis=1).astype(int)

        x_multi     = np.array(np.split(x_sig.T     , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        px_multi    = np.array(np.split(px_sig.T    , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        y_multi     = np.array(np.split(y_sig.T     , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        py_multi    = np.array(np.split(py_sig.T    , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        zeta_multi  = np.array(np.split(zeta_sig.T  , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        pzeta_multi = np.array(np.split(pzeta_sig.T , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        #========================================
        
        # Computing C-S like invariants
        Jx = 1/2 * np.mean(x_multi**2+px_multi**2,axis=1)
        Jy = 1/2 * np.mean(y_multi**2+py_multi**2,axis=1)
        Jzeta = 1/2 * np.mean(zeta_multi**2+pzeta_multi**2,axis=1)

        if self.skip_naff or (self.n_harm is None) or (self.n_harm == 0):
            # Appending to data
            #-------------------------
            self.data['turn']   = turn_idx
            self.data['torus']  = torus_idx
            self.data['state']  = state_multi
            #----------
            self.data['Jx']  = Jx
            self.data['Jy']  = Jy
            self.data['Jzeta']  = Jzeta
            #-------------------------
        else:
            # Extracting the harmonics
            #--------------------------
            n_harm       = self.n_harm
            window_order = self.window_order
            window_type  = self.window_type

            Ax,Qx       = nafflib.multiparticle_harmonics(x_multi,px_multi      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
            Ay,Qy       = nafflib.multiparticle_harmonics(y_multi,py_multi      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
            Azeta,Qzeta = nafflib.multiparticle_harmonics(zeta_multi,pzeta_multi, num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)



            # Appending to data
            #-------------------------
            self.data['turn']   = turn_idx
            self.data['torus']  = torus_idx
            self.data['state']  = state_multi
            #----------
            self.data['Ax'] = Ax
            self.data['Qx'] = Qx
            self.data['Ay'] = Ay
            self.data['Qy'] = Qy
            self.data['Azeta'] = Azeta
            self.data['Qzeta'] = Qzeta
            self.data['Jx']  = Jx
            self.data['Jy']  = Jy
            self.data['Jzeta']  = Jzeta
            #-------------------------
#===================================================












# #===================================================
# # TORUS BUFFER
# #===================================================
# class ACTION_Buffer(Buffer):
#     def __init__(self,normalize=True,complex2tuple=False):
#         super().__init__()  
#         self.clean()
#         self.normalize = normalize
#         self.complex2tuple = complex2tuple

#         # To be injected manually!
#         #=========================
#         self.twiss          = None
#         self.nemitt_x       = None
#         self.nemitt_y       = None
#         self.nemitt_zeta    = None
#         #=========================

#         # NAFF parameters
#         #=========================
#         self.n_torus      = None
#         self.n_points     = None
#         #=========================

#     def to_dict(self):
#         dct    = {}

#         for key,value in self.data.items():

#             if np.issubdtype(value[0].dtype, complex) and self.complex2tuple:
#                 # is complex
#                 dct[key] = [[(c.real, c.imag) for c in row] for row in value]
#             else:
#                 dct[key] = value.tolist()


#         return dct
        
#     def clean(self,):
#         self.data['turn']   = []
#         self.data['torus']  = []
#         self.data['state']  = []

#         self.data['Jx']  = []
#         self.data['Jy']  = []
#         self.data['Jzeta']  = []

#     def process(self,monitor):
#         self.update(monitor = monitor)

#         assert self.call_ID <= 1, "TORUS_Buffer is not designed to store multiple chunks!"


#         # Extracting data
#         #-------------------------
#         start_at_turn = monitor.start_at_turn
#         stop_at_turn  = monitor.stop_at_turn
#         self.n_turns  = stop_at_turn-start_at_turn

#         x    = monitor.x
#         px   = monitor.px
#         y    = monitor.y
#         py   = monitor.py
#         zeta = monitor.zeta
#         pzeta = monitor.pzeta

#         if self.normalize:
#             # Computing normalized coordinates
#             #--------------------------
#             XX_sig = xPlus._W_phys2norm(x,px,y,py,zeta,pzeta, 
#                                             W_matrix    = self.twiss.W_matrix,
#                                             co_dict     = self.twiss.particle_on_co.copy(_context=xo.context_default).to_dict(), 
#                                             nemitt_x    = self.nemitt_x, 
#                                             nemitt_y    = self.nemitt_y, 
#                                             nemitt_zeta = self.nemitt_zeta)

#             x_sig       = XX_sig[0,:,:]
#             px_sig      = XX_sig[1,:,:]
#             y_sig       = XX_sig[2,:,:]
#             py_sig      = XX_sig[3,:,:]
#             zeta_sig    = XX_sig[4,:,:]
#             pzeta_sig   = XX_sig[5,:,:]
#         else:
#             x_sig       = x
#             px_sig      = px
#             y_sig       = y
#             py_sig      = py
#             zeta_sig    = zeta
#             pzeta_sig   = pzeta


#         # Reshaping for faster handling
#         #========================================
#         torus_idx,turn_idx = np.mgrid[:self.n_torus,:self.n_turns]
#         torus_idx = torus_idx.reshape(self.n_torus*self.n_turns)
#         turn_idx  = turn_idx.reshape(self.n_torus*self.n_turns)
#         state_multi = np.all(np.array(np.split(monitor.state.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)==1,axis=1).astype(int)

#         x_multi     = np.array(np.split(x_sig.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
#         px_multi    = np.array(np.split(px_sig.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
#         y_multi     = np.array(np.split(y_sig.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
#         py_multi    = np.array(np.split(py_sig.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
#         zeta_multi  = np.array(np.split(zeta_sig.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
#         pzeta_multi = np.array(np.split(pzeta_sig.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
#         #========================================
        


#         Jx = 1/2 * np.mean(x_multi**2+px_multi**2,axis=1)
#         Jy = 1/2 * np.mean(y_multi**2+py_multi**2,axis=1)
#         Jzeta = 1/2 * np.mean(zeta_multi**2+pzeta_multi**2,axis=1)


#         # Appending to data
#         #-------------------------
#         self.data['turn']   = turn_idx
#         self.data['torus']  = torus_idx
#         self.data['state']  = state_multi
#         #----------
#         self.data['Jx']  = Jx
#         self.data['Jy']  = Jy
#         self.data['Jzeta']  = Jzeta
#         #-------------------------
# #===================================================