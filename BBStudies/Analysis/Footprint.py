#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed March 1 11:50:29 2021
@author: pbelange
"""


import numpy as np
import pandas as pd
import PyNAFF
import NAFFlib

     


def PyNAFF_tune(x,nterms=1,skipTurns=0):
    '''
    Compute tunes of a particle using pyNAFF
    ----------------------------------------
    Input:
        x         : x(t) of the particle (or y)
        nterms    : maximum number of harmonics to search for in the data sample
        skipTurns : number of observations (data points) to skip from the start
    Returns:
        tunes: pd.Series containing the tune in both planes
    '''
    x      = np.array(x)
    NAFF_x = PyNAFF.naff(x-np.mean(x), turns=len(x), nterms=nterms , skipTurns=skipTurns, getFullSpectrum=False)

    # TODO: allow more than 1 harmonic (nterms>1)
    # naff returns: [order of harmonic, frequency, Amplitude, Re{Amplitude}, Im{Amplitude] 
    _,Qx,_,Ax_Re,Ax_Im = NAFF_x[0]

    return Qx


def NAFFlib_tune(x,nfreqs = 1,Hann_order=2,multiparticles = False):

    

    # No need to remove average, already done.
    if multiparticles:
        x = np.array(x)
        Q = NAFFlib.multiparticle_tunes(x, order=Hann_order)

        Q[Q<0] = np.nan
        return Q

    else:
        x        = np.array(x)
        Q,Ap,An  = NAFFlib.get_tunes(x-np.mean(x), nfreqs, Hann_order)

        # np.abs(Ap[i]) is the amplitude
        if nfreqs ==1:
            return Q[0]
        else:
            return Q



def FFT_tune(x):
    '''
    Compute tune of a particle from simple FFT
    -------------------------------------------
    Input:
        x           : x(t) of the particle (or y)
        showSpectrum: {True|False} to plot the spectrum used for the fft
    Returns:
        tunes: pd.Series containing the tune in both planes
    '''

    x     = np.array(x)
    turns = np.arange(1,len(x)+1)
    freq  = np.fft.fftfreq(turns.shape[-1])


    spectrum = np.fft.fft(x-np.mean(x))
    idx      = np.argmax(np.abs(spectrum))
    Qx       = freq[idx]

    return Qx
  


