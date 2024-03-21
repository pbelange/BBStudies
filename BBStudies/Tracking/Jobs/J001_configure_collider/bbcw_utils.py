# ==================================================================================================
# --- Imports
# ==================================================================================================
import numpy as np
import xtrack as xt

import BBStudies.Tracking.Utils as xutils



bbcw_insert = {}
bbcw_insert['b1'] = {   'bbcw.t.4l1.b1':'tctpv.4l1.b1',
                        'bbcw.b.4l1.b1':'tctpv.4l1.b1',
                        'bbcw.e.4l5.b1':'tctph.4l5.b1',
                        'bbcw.i.4l5.b1':'tctph.4l5.b1'}
bbcw_insert['b2'] = {   'bbcw.t.4r1.b2':'tctpv.4r1.b2',
                        'bbcw.b.4r1.b2':'tctpv.4r1.b2',
                        'bbcw.e.4r5.b2':'tctph.4r5.b2',
                        'bbcw.i.4r5.b2':'tctph.4r5.b2'}

bbcw_xy = lambda _name: {   't':( 0, 1),
                            'b':( 0,-1),
                            'e':( 1, 0),
                            'i':(-1, 0)}[_name.split('.')[1]]

# ==================================================================================================
# --- Function to install BBCW
# ==================================================================================================
def install_BBCW(collider,L_phy=1,L_int=2):
    

    # Creating some default knobs:
    for beam_name in ["b1", "b2"]:
        for ip in ["ip1", "ip5"]:
            collider.vars[f'i_wire_{ip}.{beam_name}']   = 0 
            collider.vars[f'd_wire_{ip}.{beam_name}']   = 1
            collider.vars[f'co_y_wire_{ip}.{beam_name}']= 0
            collider.vars[f'co_x_wire_{ip}.{beam_name}']= 0

            # co correction
            collider.vars[f'kq4.l{ip[-1]}{beam_name}.k0']= 0
            collider.vars[f'kq4.r{ip[-1]}{beam_name}.k0']= 0




    # Installing in the collider
    for beam_name in ["b1", "b2"]:
        
        # Inserting all bbcw
        line = collider[f'lhc{beam_name}']
        for _bbcw,_at in bbcw_insert[beam_name].items():

            line.insert_element(name    = _bbcw,
                                at      = _at,
                                element = xt.Wire(  L_phy   = L_phy, 
                                                    L_int   = L_int,
                                                    current = 0,
                                                    xma     = bbcw_xy(_bbcw)[0], 
                                                    yma     = bbcw_xy(_bbcw)[1]))
            
            
                
            
            # Linking to the knobs
            ip = 'ip' + _bbcw.split('.')[2][-1]
            line.element_refs[_bbcw].current= collider.vars[f'i_wire_{ip}.{beam_name}']
            line.element_refs[_bbcw].xma    = bbcw_xy(_bbcw)[0]*line.vars[f'd_wire_{ip}.{beam_name}'] + collider.vars[f'co_x_wire_{ip}.{beam_name}']
            line.element_refs[_bbcw].yma    = bbcw_xy(_bbcw)[1]*line.vars[f'd_wire_{ip}.{beam_name}'] + collider.vars[f'co_y_wire_{ip}.{beam_name}']


            # Linking the co knobs as well for correction later
            for Q4_knob in [f'kq4.l{ip[-1]}{beam_name}', f'kq4.r{ip[-1]}{beam_name}']:
                slices = [t._key for t in line.vars[Q4_knob]._find_dependant_targets() if isinstance(t._key,str)]
                slices = list(set([s for s in slices if '..' in s]))
                for slice in slices:
                    if ip == 'ip1':
                        # vertical -> ksl
                        line.element_refs[slice].ksl[0] = (line.vars[Q4_knob + '.k0'] * collider.vars['l.mqy'] / len(slices))
                    else:
                        # horizontal -> knl
                        line.element_refs[slice].knl[0] = (line.vars[Q4_knob + '.k0'] * collider.vars['l.mqy'] / len(slices))

            
    


    return collider
# ==================================================================================================     


# ==================================================================================================
# --- Function to power BBCW
# ==================================================================================================
def power_BBCW(collider,config):
    config_bbcw = config['config_collider']['config_bbcw']

    # If the Q4 strenghts are forced load the file
    #=========
    if config_bbcw['qff_file'] is not None:
        force_knobs = xutils.read_metadata(config_bbcw['qff_file'])['qff_knobs']
    else:
        force_knobs = {}
    #=========
        
        
    qff_knobs = {}
    for beam_name in ['b1','b2']:
        wire_dict = config_bbcw[f'{beam_name}'] 
        
        # Skip if not powered
        #--------------------------------
        if (wire_dict['ip1']['current'] == 0) and (wire_dict['ip5']['current'] == 0): 
            continue


        # Aligning the wires:
        #--------------------------------
        line    = collider[f'lhc{beam_name}']
        tw0     = line.twiss()
        target_co = []
        for _bbcw,_at in bbcw_insert[beam_name].items():
            ip      = 'ip' + _bbcw.split('.')[2][-1]
            tw_tct  = tw0.rows[_at]

            # updating knobs
            collider.vars[f'd_wire_{ip}.{beam_name}']    = wire_dict[ip]['distance']
            collider.vars[f'co_x_wire_{ip}.{beam_name}'] = tw_tct.x
            collider.vars[f'co_y_wire_{ip}.{beam_name}'] = tw_tct.y
            target_co.append(xt.TargetSet(  x = tw_tct.x[0],
                                            y = tw_tct.y[0], at=_at, tol=1e-6,tag=f'co_wire_{ip}'))
            

        # Preparing the matcher
        #--------------------------------
        # common_targets = target_co + [  xt.TargetSet(qx=tw0.qx, qy=tw0.qy, tol=1e-6, tag='tune'),
        #                                 xt.TargetSet(betx=tw0.rows['ip1'].betx[0], bety=tw0.rows['ip1'].bety[0],at='ip1', tol=1e-6, tag='betstar_ip1'),
        #                                 xt.TargetSet(betx=tw0.rows['ip5'].betx[0], bety=tw0.rows['ip5'].bety[0],at='ip5', tol=1e-6, tag='betstar_ip5')]
        
        common_targets = target_co + [  xt.TargetSet(qx=tw0.qx, qy=tw0.qy, tol=1e-6, tag='tune') ]
        opt = {}
        for ip in ['ip1','ip5']:
            opt[ip] = line.match(solve=False,
                                    vary    = [ xt.VaryList([f'kq4.l{ip[-1]}{beam_name}', f'kq4.r{ip[-1]}{beam_name}'], step=1e-8, tag=f'quad_{ip}'),
                                                xt.VaryList([f'kq4.l{ip[-1]}{beam_name}.k0', f'kq4.r{ip[-1]}{beam_name}.k0'], step=1e-8, tag=f'quad_k0_{ip}')],
                                    targets = common_targets)
            

        # Powering the wires one-by-one:
        #-----------------------------------
        for ip in ['ip1','ip5']:
            collider.vars[f'i_wire_{ip}.{beam_name}'] = wire_dict[ip]['current']

            # assert no change on closed orbit
            ttt = opt[ip].target_status(ret=True)
            assert np.all(ttt['tol_met'][[('co_wire' in t) for t in ttt['tag']]]), 'Wires seems misaligned!'

            # If the Q4 strenghts are forced, deactivate them from the opt. and set the value
            if config_bbcw['qff_file'] is not None:
                #---------
                opt[ip].disable_vary(tag=f'quad_{ip}')
                #---------
                for _knob in [_knob.name for _knob in opt[ip].vary if _knob.tag == f'quad_{ip}']:
                    collider.vars[_knob] = force_knobs[beam_name][_knob]

                
            # Matching
            opt[ip].solve()

    qff_knobs[beam_name] = {**opt['ip1'].get_knob_values(), **opt['ip5'].get_knob_values()}
    return collider,qff_knobs
