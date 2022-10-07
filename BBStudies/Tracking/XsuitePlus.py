
import json
import rich
import re
import numpy as np
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn,TimeElapsedColumn,SpinnerColumn,TimeRemainingColumn

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
def importLine(fname):
    
    with open(fname, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data)
    line.particle_ref = xp.Particles.from_dict(input_data['particle_on_tracker_co'])
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



# Tracking class:
#===================================================
class Tracking():
    
    def __init__(self,tracker,particles,n_turns,progress=False,saveVars = False):
        
        self.particles = particles.copy()
        self.n_turns   = int(n_turns)
        self.vars      = None

        # Footprint info
        self._tunes    = None
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
        try:
            self.runTracking(tracker)
        except KeyboardInterrupt:
            self.closeLiveDisplay()

        # Disabling Tracking
        self.runTracking = lambda _: print('New Tracking instance needed')
    


    @property
    def tunes(self):
        # Reset if method is changed
        if self._tunesMTD != self._oldTunesMTD:
            self._tunes = None

        if self._tunes is None:
            if self._tunesMTD == 'pynaff':
                self._oldTunesMTD = 'pynaff'
                self._tunes    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.NAFF_tune(_part['x']),'Qy':footp.NAFF_tune(_part['y'])}))
            if self._tunesMTD == 'fft':
                self._oldTunesMTD = 'fft'
                self._tunes    = self.df.groupby('particle').apply(lambda _part: pd.Series({'Qx':footp.FFT_tune(_part['x']),'Qy':footp.FFT_tune(_part['y'])}))

        return self._tunes


    def runTracking(self,tracker):
        if self.progress:
            # Create monitor if needed
            if self.monitor is None:
                self.monitor = xt.ParticlesMonitor( start_at_turn    = 0, 
                                                    stop_at_turn     = self.n_turns,
                                                    n_repetitions    = 1,
                                                    repetition_period= 1,
                                                    num_particles    = len(self.particles.particle_id))

            # Run turn-by-turn to show progress
            self.startProgressBar()
            #-------------------------
            for iturn in range(self.n_turns):
                self.monitor.track(self.particles)
                tracker.track(self.particles)
                self.updateProgressBar()
            #-------------------------
            self.closeLiveDisplay()

            #CONVERT TO PANDAS
            self.df = pd.DataFrame(self.monitor.to_dict()['data'])

        else:
            self.startSpinner()
            tracker.track(self.particles, num_turns=self.n_turns,turn_by_turn_monitor=True)
            self.closeLiveDisplay()

            #CONVERT TO PANDAS
            self.df = pd.DataFrame(tracker.record_last_track.to_dict()['data'])
        
        # Filter the data
        self.df = self.df[['at_turn','particle_id','x','px','y','py','zeta','delta','state']]
        self.df.rename(columns={"at_turn": "turn",'particle_id':'particle'},inplace=True)

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