import numpy as np
import pandas as pd
from pathlib import Path
import gc
import scipy.special as sciSpec

# BOKEH
import bokeh.plotting as bk
import bokeh.models as bkmod
import bokeh.layouts as bklay
    
# xsuite
import xtrack as xt
import xmask as xm
import xfields as xf
import xpart as xp

# BBStudies
import sys
sys.path.append('/Users/pbelanger/ABPLocal/BBStudies')
import BBStudies.Physics.Detuning as tune
import BBStudies.Physics.Constants as cst
    

# Save to HTML
#=====================================
def export_HTML(LAYOUT,filename,tabname):

    bk.output_file(filename=filename, title=tabname)
    bk.save(LAYOUT)

    print(f'Saved {tabname}:{filename}')
#======================================



# New axis function
#=====================================
def new_y_axis(fig,axis_name,side='none'):
    fig.extra_y_ranges[axis_name] = bkmod.Range1d(0,1)
    _ax = bkmod.LinearAxis(y_range_name=axis_name)
    if side == 'none':
        pass
    else:
        fig.add_layout(_ax,side)

    return _ax,axis_name
#-------------------------------------
def new_x_axis(fig,axis_name,side='none'):
    fig.extra_x_ranges[axis_name] = bkmod.Range1d(0,1)
    _ax = bkmod.LinearAxis(x_range_name=axis_name)
    if side == 'none':
        pass
    else:
        fig.add_layout(_ax,side)

    return _ax,axis_name
#======================================



def extract_lattice_info(line,twiss):

    # Finding magnets by _order component of Mulitpole
    _type_dict = {'0':'dipole','1':'quadrupole','2':'sextupole','3':'octupole'}


    # Creating lattice dictionary
    lattice =  {}
    lattice['name']  = []    
    lattice['type']  = []
    lattice['length']= []
    lattice['s']     = []
    lattice['knl']   = []
    lattice['ksl']   = []
    
    # Iterating through the elements
    all_multipoles = line.get_elements_of_type(xt.elements.Multipole)
    tw_data        = twiss.set_index('name').loc[all_multipoles[1],['s','x','y','betx','bety']]

    # for ee,name,s in zip(*all_multipoles,s_values):
    for ee,(name,tw_row) in zip(all_multipoles[0],tw_data.iterrows()):
        
        if ee._order>3:
            continue

        if ee._order==0:
            if 'mb' not in name.split('.')[0]:
                continue

        lattice['name'].append(name)
        lattice['type'].append(_type_dict[str(ee._order)])
        lattice['length'].append(ee.length)
        lattice['s'].append(tw_row.s)
        lattice['knl'].append(ee.knl[ee._order])
        lattice['ksl'].append(ee.ksl[ee._order])

    lattice_df = pd.DataFrame(lattice)
    unsliced   = lattice_df.groupby(lattice_df.name.apply(lambda name: name.split('..')[0]))

    light_lattice = {}
    light_lattice['name']    = []    
    light_lattice['type']    = []
    light_lattice['length']  = []
    light_lattice['s']       = []
    light_lattice['s_entry'] = []
    light_lattice['s_exit']  = []
    light_lattice['s']       = []
    light_lattice['knl']     = []
    light_lattice['ksl']     = []
    for name,group in unsliced:
        light_lattice['name'].append(name)
        light_lattice['type'].append(group.type.values[0])
        light_lattice['length'].append(group.length.sum())
        light_lattice['s'].append(group.s.min()+group.length.sum()/2)
        light_lattice['s_entry'].append(group.s.min())
        light_lattice['s_exit'].append(group.s.min()+group.length.sum())
        light_lattice['knl'].append(group.knl.mean())
        light_lattice['ksl'].append(group.ksl.mean())


    return pd.DataFrame(light_lattice).sort_values(by='s').reset_index(drop=True)






def bblr_knl(ee_bb,dx,dy):
    Nb    = ee_bb.n_particles
    IL_eq = Nb*cst.elec*cst.c

    gamma0 = 1/np.sqrt(1-ee_bb.beta0**2) 
    E      = gamma0*xp.PROTON_MASS_EV
    p0     = ee_bb.beta0*E/cst.c

    
    n = np.arange(12+1)
    integratedComp = -cst.mu0*(IL_eq)*sciSpec.factorial(n)/(2*np.pi)/(dx+1j*dy)**(n+1)
    _kn,_sn = np.real(integratedComp),np.imag(integratedComp)
    
    knl,snl = _kn/p0,_sn/p0
    return  knl,snl


def compute_bblr_strength(ee_bb,x,y,betx,bety,flip_x_coord=False):
    # Fixing exmittance since strength should be normalized at the end
    emittxy = [1,1]

    # Computing beam separation
    if flip_x_coord:
        x = -x
    dx = x - ee_bb.ref_shift_x - ee_bb.other_beam_shift_x
    dy = y - ee_bb.ref_shift_y - ee_bb.other_beam_shift_y

    # Computing knl of interaction
    knl,snl = bblr_knl(ee_bb,dx,dy)
    
    # Computing tune shift
    vec_J = {}
    for plane,amplitudes in zip(['x','y'],[[1,0],[0,1]]):
        ax,ay = amplitudes
        DQx,DQy = tune.DQx_DQy_octupole(ax,ay,  betxy   = [betx,bety],
                                                emittxy = emittxy,
                                                k1l     = 0,
                                                k3l     = knl[3])
    
        vec_J[plane] = np.array([DQx,DQy])


    # Strength
    #---------------------------------------
    area   = np.abs(np.cross(list(vec_J['x']), list(vec_J['y']))/2)
    len_x = np.linalg.norm(vec_J['x'])
    len_y = np.linalg.norm(vec_J['y'])
    # strength = area
    #---------------------------------------

    # return strength*ee_bb.scale_strength
    return len_x*ee_bb.scale_strength,len_y*ee_bb.scale_strength



def extract_bblr_info(line,twiss):

    # Creating lattice dictionary
    lattice =  {}
    lattice['name']    = []    
    lattice['type']    = []
    lattice['length']  = []
    lattice['s']       = []
    lattice['s_entry'] = []
    lattice['s_exit']  = []
    lattice['strength_x']= []
    lattice['strength_y']= []

    # Iterating through the elements
    all_bblr = line.get_elements_of_type(xf.beam_elements.beambeam2d.BeamBeamBiGaussian2D)
    tw_data  = twiss.set_index('name').loc[all_bblr[1],['s','x','y','betx','bety']]

    # Finding if we need to flip x axis or not
    beam     = all_bblr[1][0].split('_')[1][-2:]
    if beam.lower() == 'b2':
        flip_x_coord = True
    else:
        flip_x_coord = False

    for ee,(name,tw_row) in zip(all_bblr[0],tw_data.iterrows()):
        # print(name,ee.scale_strength,tw_row.s)

        strength_x,strength_y = compute_bblr_strength(ee,tw_row.x,tw_row.y,tw_row.betx,tw_row.bety,flip_x_coord=flip_x_coord)
        length   = 7.5/10
        lattice['name'].append(name)
        lattice['type'].append('bblr')
        lattice['length'].append(length)
        lattice['s'].append(tw_row.s + length/2)
        lattice['s_entry'].append(tw_row.s)
        lattice['s_exit'].append(tw_row.s + length)
        lattice['strength_x'].append(strength_x)
        lattice['strength_y'].append(strength_y)

    return pd.DataFrame(lattice).sort_values(by='s').reset_index(drop=True)



def extract_collimation_info(line,twiss):


    # Defining collimator types
    _type_dict     = {'tcp.':'Primary','tcsg.':'Secondary','tcla.':'Absorber','tctp':'Tertiary','bbcw':'BBCW'}
    _length_dict   = {'Primary': 0.6, 'Secondary': 1.0, 'Absorber':1.0,'Tertiary': 1.0,'BBCW':1.0} 
    _strength_dict = {'Primary': 1, 'Secondary': 0.75, 'Absorber':0.5,'Tertiary': 0.5,'BBCW':0.2} # will be -0.05 later


    # Creating lattice dictionary
    lattice =  {}
    lattice['name']    = []    
    lattice['type']    = []
    lattice['length']  = []
    lattice['s']       = []
    lattice['s_entry'] = []
    lattice['s_exit']  = []
    lattice['strength']= []

    # Iterating through the elements
    all_colls = []
    for _type in _type_dict.keys():
        all_colls += [name for name in twiss.name if _type in name]

    tw_data        = twiss.set_index('name').loc[all_colls,['s']].reset_index()

    # for ee,name,s in zip(*all_multipoles,s_values):
    for (name,tw_group) in tw_data.groupby(tw_data.name.apply(lambda name:name.split('_')[0])):
        
        _type   = [_type_dict[key] for key in _type_dict.keys() if key in name][0]
        _length = _length_dict[_type]

        lattice['name'].append(name)
        lattice['type'].append(_type)
        lattice['length'].append(_length)
        lattice['s'].append(tw_group.s.min()+_length/2)
        lattice['s_entry'].append(tw_group.s.min())
        lattice['s_exit'].append(tw_group.s.min()+_length)
        lattice['strength'].append(_strength_dict[_type]-0.05)

    return pd.DataFrame(lattice).sort_values(by='s').reset_index(drop=True)
