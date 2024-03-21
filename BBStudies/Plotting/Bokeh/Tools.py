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
import BBStudies.Physics.Detuning as tune
import BBStudies.Physics.Constants as cst
    

# Save to HTML
#=====================================
def export_HTML(LAYOUT,filename,tabname='Bokeh - Figure'):

    bk.output_file(filename=filename, title=tabname)
    bk.save(LAYOUT)

    print(f'Saved {tabname}:{filename}')
#======================================


# From chatGPT
def dict_to_HTML(my_dict, header_text="Dictionary Content", header_margin_top=10, header_margin_bottom=20, margin=200, indent=10, nested_scale=0.5, max_width=800):
    def generate_html_content(data, is_nested=False, margin=margin, indent=indent, nested_scale=nested_scale, max_width=max_width):
        _width = f'{nested_scale * 100:.0f}%'
        html_content = ""

        # Determine the class based on whether it's a nested dictionary or not
        class_name = "nested-attribute" if is_nested else "attribute"

        # Create a common box for all attributes at the first level
        if not is_nested:
            width_style = f"width: {_width}px;"
            max_width_style = f"max-width: {max_width}px;" if max_width is not None else ""
            html_content += f"""
            <div class="{class_name}" style="margin-left: {margin}px; {width_style} {max_width_style}; font-family: 'Courier New', monospace;">
        """

        # Calculate the maximum key length
        max_key_length = max([len(key) for key in data.keys()])

        for key, value in data.items():
            # Pad the key to ensure the same length
            padded_key = f"{key}:".ljust(max_key_length + 1)

            html_content += f"""
                <p><strong style="white-space: pre;">{padded_key}</strong> 
            """
            if isinstance(value, dict):
                # Display nested dictionary content in a new grey box
                nested_max_width_style = f"width: {_width};"
                if max_width is not None:
                    nested_max_width_style += f" max-width: {max_width}px;"

                html_content += f"""
                <div class="nested-attribute" style="margin-left: {indent}px; {nested_max_width_style}; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 5px;">
                    {generate_html_content(value, is_nested=True, margin=margin, indent=indent, nested_scale=nested_scale, max_width=max_width)}
                </div>
                """
            else:
                html_content += f"{value}</p>"

        # Close the common box for all attributes at the first level
        if not is_nested:
            html_content += """
            </div>
        """

        return html_content

    # Create HTML content from the dictionary
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 10px;
            }}

            #header {{
                font-size: 20px;
                font-weight: bold;
                margin-top: {header_margin_top}px;
                margin-bottom: {header_margin_bottom}px;
                margin-left: {margin}px;  /* Set left margin */
            }}

            .attribute, .nested-attribute {{
                margin-bottom: 10px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                width: {max_width}px;  /* Set static width */
                font-family: 'Courier New', monospace;
            }}

            .attribute h3, .nested-attribute h3 {{
                color: #333;
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <div id="header">{header_text}</div>
    """ + generate_html_content(my_dict, margin=margin, indent=indent, nested_scale=nested_scale, max_width=max_width) + """
    </body>
    </html>
    """

    # Create a Bokeh Div widget
    return bkmod.Div(text=html_content)



# Source from groupby to facilite sliders
#=========================================
def source_from_groupby(df,by,columns):
    _df_list = []
    for col in columns:
        _df = pd.DataFrame({f'{col}:{_key}':_group[col].values for _key,_group in df.groupby(by)})
        _df_list.append(_df)

    to_source = pd.concat(_df_list,axis=1)
    for col in columns[::-1]:
        to_source.insert(0,f'{col}:active',to_source[f'{col}:0'])
    return to_source
#=========================================
# Set aspect ratio of the fig.
#=====================================
def set_aspect(fig, x_lim,y_lim, aspect=1, margin=0):
    """Set the plot ranges to achieve a given aspect ratio.
    """

    xmin,xmax = x_lim
    ymin,ymax = y_lim

    width = (xmax - xmin)#*(1+2*margin)
    if width <= 0:
        width = 1.0
    height = (ymax - ymin)#*(1+2*margin)
    if height <= 0:
        height = 1.0
    xcenter = 0.5*(xmax + xmin)
    ycenter = 0.5*(ymax + ymin)
    r = aspect*((fig.width-margin)/fig.height)
    # if width < r*height:
        # width = r*height
    # else:
    height = width/r
    fig.x_range = bkmod.Range1d(xcenter-0.5*width, xcenter+0.5*width)
    fig.y_range = bkmod.Range1d(ycenter-0.5*height, ycenter+0.5*height)
#=====================================


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

#======================================
def excursion_polygon(row):
    # skew col : y = ax + b
    skew_angle = 127.5
    a = np.tan(np.deg2rad(skew_angle-90))
    b = np.max([np.abs(row['skew_max']),np.abs(row['skew_min'])])/np.cos(np.deg2rad(skew_angle-90))

    # skew col : y = ax - b, x = (y+b)/a
    x1 = [row['x_max'],a*row['x_max'] - b]
    x2 = [(row['y_min']+b)/a,row['y_min']]

    # skew col : y = -ax - b, x = -(y+b)/a
    x3 = [-(row['y_min']+b)/a,row['y_min']]
    x4 = [row['x_min'],-a*row['x_min'] - b]

    # skew col : y = ax + b, x = (y-b)/a
    x5 = [row['x_min'],a*row['x_min'] + b]
    x6 = [(row['y_max']-b)/a,row['y_max']]

    # skew col : y = -ax + b, x = -(y-b)/a
    x7 = [-(row['y_max']-b)/a,row['y_max']]
    x8 = [row['x_max'],-a*row['x_max'] + b]

    if x5[1]<x4[1]:
        x4[1],x5[1] = x5[1],x4[1]
    
    if x8[1]<x1[1]:
        x1[1],x8[1] = x8[1],x1[1]

    return [x1[0],x2[0],x3[0],x4[0],x5[0],x6[0],x7[0],x8[0]], [x1[1],x2[1],x3[1],x4[1],x5[1],x6[1],x7[1],x8[1]]
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

def compute_bblr_strength_kick(ee_bb,x,y,betx,bety,flip_x_coord=False):
    # Fixing exmittance since strength should be normalized at the end
    emittxy = [1,1]

    # Computing beam separation
    if flip_x_coord:
        x = -x
    dx = x - ee_bb.ref_shift_x - ee_bb.other_beam_shift_x
    dy = y - ee_bb.ref_shift_y - ee_bb.other_beam_shift_y
    Nb    = ee_bb.n_particles
    IL_eq = Nb*cst.elec*cst.c

    gamma0 = 1/np.sqrt(1-ee_bb.beta0**2)
    
    r     = np.sqrt(x**2+y**2)
    sig_x = np.sqrt(betx*emittxy[0])
    sig_y = np.sqrt(bety*emittxy[1])

    kick_factor = 2*cst.r_p*Nb/gamma0
    kick_x = kick_factor*x*(1-np.exp(-r**2/(2*sig_x**2)))/r**2
    kick_y = kick_factor*y*(1-np.exp(-r**2/(2*sig_y**2)))/r**2

    # return strength*ee_bb.scale_strength
    return kick_x*ee_bb.scale_strength,kick_y*ee_bb.scale_strength


def compute_bbho_strength(ee_bb,x,y,betx,bety,beta0,flip_x_coord=False):
    # Fixing exmittance since strength should be normalized at the end
    emittxy = [1,1]

    # Computing beam separation
    if flip_x_coord:
        x = -x
    dx = x - ee_bb.ref_shift_x - ee_bb.other_beam_shift_x
    dy = y - ee_bb.ref_shift_y - ee_bb.other_beam_shift_y

    Nb     = ee_bb.slices_other_beam_num_particles[0]
    gamma0 = 1/np.sqrt(1-beta0**2) 

    r     = np.sqrt(x**2+y**2)
    sig_x = np.sqrt(betx*emittxy[0])
    sig_y = np.sqrt(bety*emittxy[1])

    kick_factor = 2*cst.r_p*Nb/gamma0
    kick_x = kick_factor*x*(1-np.exp(-r**2/(2*sig_x**2)))/r**2
    kick_y = kick_factor*y*(1-np.exp(-r**2/(2*sig_y**2)))/r**2

    # return strength*ee_bb.scale_strength
    return kick_x*ee_bb.scale_strength,kick_y*ee_bb.scale_strength


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
    # if beam.lower() == 'b2':
        # flip_x_coord = True
    # else:
    flip_x_coord = False

    for ee,(name,tw_row) in zip(all_bblr[0],tw_data.iterrows()):
        # print(name,ee.scale_strength,tw_row.s)

        # strength_x,strength_y = compute_bblr_strength(ee,tw_row.x,tw_row.y,tw_row.betx,tw_row.bety,flip_x_coord=flip_x_coord)
        strength_x,strength_y = compute_bblr_strength_kick(ee,tw_row.x,tw_row.y,tw_row.betx,tw_row.bety,flip_x_coord=flip_x_coord)
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


def extract_bbho_info(line,twiss):

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
    all_bblr = line.get_elements_of_type(xf.beam_elements.beambeam3d.BeamBeamBiGaussian3D)
    tw_data  = twiss.set_index('name').loc[all_bblr[1],['s','x','y','betx','bety']]

    # Finding if we need to flip x axis or not
    beam     = all_bblr[1][0].split('_')[1][-2:]
    # if beam.lower() == 'b2':
        # flip_x_coord = True
    # else:
    flip_x_coord = False

    for ee,(name,tw_row) in zip(all_bblr[0],tw_data.iterrows()):
        # print(name,ee.scale_strength,tw_row.s)

        strength_x,strength_y = compute_bbho_strength(ee,tw_row.x,tw_row.y,tw_row.betx,tw_row.bety,line.particle_ref.beta0[0],flip_x_coord=flip_x_coord)
        length   = 0.005
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
