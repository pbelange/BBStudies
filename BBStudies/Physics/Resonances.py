

import pandas as pd
import numpy as np
from fractions import Fraction


# See Numerical Methods 2018 CAS proceedings by Rogelio Tomás:
#  https://cds.cern.ch/record/2723239/files/CERN-ACC-2020-0012.pdf




def Farey(order):
    # Create array of (value,numerator,denominator)
    allFracts  = [(0,0,1)] + [(m/k,)+Fraction(f'{m}/{k}').as_integer_ratio() for k in range(1, order+1) for m in range(1, k+1)]
    uniqueFraq = np.array(list(set(allFracts)))
    # sort by value (1st columns) and return
    return list(uniqueFraq[np.argsort(uniqueFraq[:,0]),1:].astype(int))


def resonance_df(order):
    ''' Returns a dataframe with the information of resonance lines up to a given order'''
    resonances = []
    nodes      = Farey(int(order))
    for node_i in nodes:
        h, k = node_i            # Node h/k on the axes
        for node_j in nodes:
            p, q = node_j
            b    = float(k * p)  # Resonance line a*Qx + b*Qy = c (linked to p/q)
            if b > 0:
                a, c = float(q - k * p), float(p * h)

                # Resonance lines from a*Qx + b*Qy = c
                # Qy = c/b - a/b*Qx
                # Qy = c/b + a/b*Qx
                # Qx = c/b - a/b*Qy     -> Qy = -(Qx - c/b)*b/a      if a!=0 else Qx = c/b, Qy = [0,1]
                # Qx = c/b + a/b*Qy     -> Qy =  (Qx - c/b)*b/a      if a!=0 else Qx = c/b, Qy = [0,1]
                # Qx = c/b - a/b*(1-Qy) -> Qy =  (Qx - (c-a)/b)*b/a  if a!=0 else Qx = c/b, Qy = [0,1]
                # Qx = c/b + a/b*(1-Qy) -> Qy = -(Qx - (c+a)/b)*b/a  if a!=0 else Qx = c/b, Qy = [0,1]

                if a!=0 :
                    slopes = [-a/b  , a/b, -b/a,  b/a,      b/a,     -b/a]
                    y0s    = [ c/b  , c/b,  c/a, -c/a, -(c-a)/a,  (c+a)/a]
                    x0s    = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan  ]
                else:
                    slopes = [-a/b  ,   a/b, np.inf]
                    y0s    = [ c/b  ,   c/b, np.nan]
                    x0s    = [np.nan,np.nan,    c/b]

                for slope,y0,x0 in zip(slopes,y0s,x0s):
                    
                    # Create unique ID to later eliminate duplicates
                    if slope != np.inf:
                        slope_int = Fraction(str(slope)).limit_denominator(20).as_integer_ratio()
                        y0_int    = Fraction(str(y0)).limit_denominator(20).as_integer_ratio()
                        ID        = slope_int+y0_int
                    else:
                        ID        = (np.inf,np.inf) + Fraction(str(x0)).limit_denominator(20).as_integer_ratio()
                    
                    resonances.append({ 'ID'    :ID,
                                        'Order' :int(a+b),
                                        'a'     :int(a),
                                        'b'     :int(b),
                                        'hkpq'  : (int(h),int(k),int(p),int(q)), 
                                        'slope' :slope,
                                        'y0'    :y0,
                                        'x0'    :x0})

            if q == k and p == 1: 
                break

    resonances = pd.DataFrame(resonances)
    # resonances = resonances.drop_duplicates(subset='ID').reset_index(drop=True)
    return resonances


