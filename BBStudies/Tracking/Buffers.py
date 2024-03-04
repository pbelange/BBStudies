import numpy as np
import pandas as pd

import nafflib
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
class CPU_monitor:
    
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
        
        self.x = monitor.x
        self.px = monitor.px
        self.y = monitor.y
        self.py = monitor.py
        self.zeta = monitor.zeta
        self.pzeta = monitor.pzeta
        self.state = monitor.state
        self.start_at_turn = monitor.start_at_turn
        self.stop_at_turn = monitor.stop_at_turn
        self.part_id_start = monitor.part_id_start
        self.part_id_end = monitor.part_id_end
#===================================================


#===================================================
# Storage monitor
# Class to store data over several chunks
#===================================================
class storage_monitor:
    
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

        self.call_ID = None
        
    def process(self,monitor):
        if self.call_ID is None:
            self.x = monitor.x
            self.px = monitor.px
            self.y = monitor.y
            self.py = monitor.py
            self.zeta = monitor.zeta
            self.pzeta = monitor.pzeta
            self.state = monitor.state[:,-1]
            self.start_at_turn = monitor.start_at_turn
            self.stop_at_turn = monitor.stop_at_turn
            self.part_id_start = monitor.part_id_start
            self.part_id_end = monitor.part_id_end
        else:
            assert self.part_id_start==monitor.part_id_start, "part_id_start should be the same"
            assert self.part_id_end==monitor.part_id_end, "part_id_end should be the same"
            
            # concatenate data:
            self.x = np.concatenate((self.x,monitor.x),axis=1)
            self.px = np.concatenate((self.px,monitor.px),axis=1)
            self.y = np.concatenate((self.y,monitor.y),axis=1)
            self.py = np.concatenate((self.py,monitor.py),axis=1)
            self.zeta = np.concatenate((self.zeta,monitor.zeta),axis=1)
            self.pzeta = np.concatenate((self.pzeta,monitor.pzeta),axis=1)
            self.state = np.concatenate((self.state,monitor.state[:,-1]),axis=1)
            
            # Keeping start_at_turn, updating stop_at_turn
            self.stop_at_turn = monitor.stop_at_turn


#===================================================


#===================================================
# CHECKPOINT BUFFER
#===================================================
class Checkpoint_Buffer(Buffer):
    def __init__(self,):

        super().__init__()

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
        


    def process(self,monitor):
        self.update(monitor = monitor)


        start_at_turn = monitor.start_at_turn

        # Appending to data
        #-------------------------
        self.data['Chunk ID'].append(self.call_ID)
        self.data['turn'].append(start_at_turn)
        self.data['particle'].append(self.particle_id)
        self.data['state'].append(monitor.state[:,0])
        self.data['x'].append(monitor.x[:,0])
        self.data['px'].append(monitor.px[:,0])
        self.data['y'].append(monitor.y[:,0])
        self.data['py'].append(monitor.py[:,0])
        self.data['zeta'].append(monitor.zeta[:,0])
        self.data['pzeta'].append(monitor.pzeta[:,0])
        #-------------------------
#===================================================
        
#===================================================
# EXCURSION BUFFER
#===================================================
class Excursion_Buffer(Buffer):
    def __init__(self,):
        
        super().__init__()  

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

        


    def process(self,monitor):
        self.update(monitor = monitor)


        # Extracting data
        #-------------------------
        start_at_turn = monitor.start_at_turn
        stop_at_turn  = monitor.stop_at_turn

        x    = monitor.x
        y    = monitor.y
         
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


        # Skew ----------
        skew_max = np.max(x_skew,axis=1)
        skew_min = np.min(x_skew,axis=1)


        # Appending to data
        #-------------------------
        self.data['Chunk ID'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['state'].append(monitor.state[:,-1])
        self.data['start_at_turn'].append(start_at_turn)
        self.data['stop_at_turn'].append(stop_at_turn)
        self.data['x_min'].append(x_min)
        self.data['x_max'].append(x_max)
        self.data['y_min'].append(y_min)
        self.data['y_max'].append(y_max)
        self.data['skew_min'].append(skew_min)
        self.data['skew_max'].append(skew_max)
        #-------------------------
#===================================================
        



#===================================================
# NAFF BUFFER
#===================================================
class NAFF_Buffer(Buffer):
    def __init__(self,):

        super().__init__()  

        # To be injected manually!
        #=========================
        self.W_matrix       = None
        self.particle_on_co = None
        self.nemitt_x       = None
        self.nemitt_y       = None
        self.nemitt_zeta    = None
        #=========================

        # NAFF parameters
        #=========================
        self.n_harm       = None
        self.window_order = None
        self.window_type  = None
        #=========================

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


        # Computing normalized coordinates
        #--------------------------
        XX_n   = [xPlus.W_phys2norm(_x,_px,_y,_py,_zeta,_pzeta,W_matrix=self.W_matrix,particle_on_co=self.particle_on_co) for (_x,_px,_y,_py,_zeta,_pzeta) in zip(x,px,y,py,zeta,pzeta)]
        XX_sig = [xPlus.norm2sigma(_XX_n[0,:],_XX_n[1,:],_XX_n[2,:],_XX_n[3,:],_XX_n[4,:],_XX_n[5,:],nemitt_x= self.nemitt_x, nemitt_y= self.nemitt_y, nemitt_zeta= self.nemitt_zeta, particle_on_co=self.particle_on_co) for _XX_n in XX_n]

        x_sig       = [_XX_sig[0,:] for _XX_sig in XX_sig]
        px_sig      = [_XX_sig[1,:] for _XX_sig in XX_sig]
        y_sig       = [_XX_sig[2,:] for _XX_sig in XX_sig]
        py_sig      = [_XX_sig[3,:] for _XX_sig in XX_sig]
        zeta_sig    = [_XX_sig[4,:] for _XX_sig in XX_sig]
        pzeta_sig   = [_XX_sig[5,:] for _XX_sig in XX_sig]


        # Extracting the harmonics
        #--------------------------
        n_harm       = self.n_harm
        window_order = self.window_order
        window_type  = self.window_type
        try:
            Ax,Qx       = nafflib.multiparticle_harmonics(x_sig,px_sig, num_harmonics=n_harm, window_order=window_order, window_type=window_type)
            Ay,Qy       = nafflib.multiparticle_harmonics(y_sig,py_sig, num_harmonics=n_harm, window_order=window_order, window_type=window_type)
            Azeta,Qzeta = nafflib.multiparticle_harmonics(zeta_sig,pzeta_sig, num_harmonics=n_harm, window_order=window_order, window_type=window_type)
        except Exception as error:
            print("An exception occurred:", type(error).__name__, "-", error) # An exception occurred
            n_part = len(x)
            Ax,Qx = np.array(n_part * [n_harm*[np.nan+ 1j*np.nan]]),np.array(n_part * [n_harm*[np.nan]])
            Ay,Qy =  np.array(n_part * [n_harm*[np.nan+ 1j*np.nan]]),np.array(n_part * [n_harm*[np.nan]])
            Azeta,Qzeta =  np.array(n_part * [n_harm*[np.nan+ 1j*np.nan]]),np.array(n_part * [n_harm*[np.nan]])


        # Appending to data
        #-------------------------
        self.data['Chunk ID'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['state'].append(monitor.state[:,-1])
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
#===================================================