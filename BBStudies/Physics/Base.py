import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


import BBStudies.Physics.Constants as cst


##############################################################
def hypersphere(N, D, r=1, seed = 0, surface=False,unpack = False):
    # Set the random seed for reproducibility
    rng = np.random.default_rng(int(seed))

    # Sample D vectors of N Gaussian coordinates
    N = int(N)
    D = int(D)
    samples = rng.standard_normal(size = (N, D))
    
    # Normalise all distances (radii) to 1
    radii = np.sqrt(np.sum(samples ** 2, axis=1))[:,np.newaxis]
    samples = samples / radii
    
    # Sample N radii with exponential distribution (unless points are to be on the surface)
    if not surface:
        new_radii = np.random.uniform(low=0.0, high=1.0, size=(N, 1)) ** (1 / D)
        samples = samples * new_radii
    
    # Scale the samples to the desired radius
    if isinstance(r,list):
        r = np.array(r)[np.newaxis,:]
    elif isinstance(r,type(np.array([]))):
        assert False, 'r should be float or list'
    samples = samples * r
    
    if not unpack:
        return samples
    else:
        return samples.T
##############################################################
    


##############################################################
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)
##############################################################


##############################################################
def vecNorm(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)
##############################################################

##############################################################
def resample(x,y,num_points):
    # compute the distances, ds, between points
    dx, dy = x[+1:]-x[:-1],  y[+1:]-y[:-1]
    ds = np.array((0, *np.sqrt(dx*dx+dy*dy)))

    # compute the total distance from the 1st point, measured on the curve
    s = np.cumsum(ds)

    # interpolate using 200 point
    xinter = np.interp(np.linspace(0,s[-1], num_points), s, x)
    yinter = np.interp(np.linspace(0,s[-1], num_points), s, y)
    return xinter, yinter
##############################################################


##############################################################
def smooth_step_function(x, step_location, step_height, step_width):
    return step_height / (1 + np.exp(-2 * (x - step_location) / step_width))

def radial_steps(r_min=1,r_max=10,num_steps = 10,smooth_factor = 1,num_points = 1000):

    step_size  = (r_max-r_min)/(num_steps-1)
    step_width = num_points/200*smooth_factor 

    x = np.arange(num_points)
    step_locations = np.linspace(np.min(x), np.max(x), num_steps+1)[1:-1]

    # Apply steps at specified locations
    y = r_min + np.zeros(len(x))
    for location in step_locations:
        y += smooth_step_function(x, location, step_size, step_width)

    return y

def serpentine_path(r_min=1,r_max=10,theta_min=0,theta_max=np.pi/2,r_offset=0,num_steps = 10,smooth_factor = 1,num_points = 1000):
    r       = radial_steps(r_min=r_min-r_offset,r_max=r_max-r_offset,num_steps=num_steps,smooth_factor=smooth_factor,num_points=num_points)
    A       = (theta_max-theta_min)/2
    y0      = (theta_max+theta_min)/2
    theta   = A * np.cos(0.5*np.linspace(0, num_steps*2*np.pi, len(r))) + y0

    x = (r) * np.cos(theta) + r_offset*np.cos(np.pi/4)
    y = (r) * np.sin(theta) + r_offset*np.sin(np.pi/4)
    return x ,y 
##############################################################


##############################################################
def phys2norm(x,px,alpha=None,beta=None,SVD=False):
    
    if SVD:
        alpha,beta = SVD_AlphaBeta(x,px)
        
    #N0 = [[1/np.sqrt(beta),0],[alpha/np.sqrt(beta), np.sqrt(beta)]]
    x_n  = x/np.sqrt(beta)
    px_n = alpha*x/np.sqrt(beta) + px*np.sqrt(beta)
    
    return x_n,px_n

def norm2phys(x_n,px_n,alpha=None,beta=None):
    
    
    x    = x_n*np.sqrt(beta)
    px   = -alpha*x_n/np.sqrt(beta) + px_n/np.sqrt(beta)
    
    return x, px
##############################################################


##############################################################
def SVD_AlphaBeta(x,px):
    '''Taken from https://arxiv.org/pdf/2006.10661.pdf '''
    
    U,s,V= np.linalg.svd([x,px])         #SVD
    
    N = np.dot(U,np.diag(s))
    theta = np.arctan(-N[0,1]/N[0,0])    #AngleofR(theta)
    co=np.cos(theta) ; si=np.sin(theta)
    
    R = [[co,si],[-si,co]]   
    X = np.dot(N,R)                      #Floquetupto1/det(USR)
    
    beta = np.abs(X[0,0]/X[1,1])
    alpha = X[1,0]/X[1,1]
    
    # dropped
    ex =s[0]*s[1]/(len(x)/2.)            #emit=det(S)/(n/2)
    
    return alpha,beta

##############################################################


##############################################################
def getAction(x,px,alpha=None,beta=None,SVD=False):
    
    if SVD:
        alpha,beta = SVD_AlphaBeta(x,px)
    gamma = (1+alpha**2)/beta
    
    J = (gamma*x**2  + 2*alpha*x*px + beta*px**2)/2
    
    return J
##############################################################


##############################################################
def generate_coordGrid(xRange,yRange,labels = ['x','y'],nPoints=100):
    '''
    Distribute points uniformly on a 2D grid.
    -----------------------------------------
    Input:
        xRange : range of first coordinate
        yRange : range of second coordinate
        labels : labels to be used in the resulting dataframe
        nPoint : total number of points to generate (sqrt(nPoints) for each coordinate)
    Returns:
        coordinates: dataframe containing the distributed points
    '''

    if type(xRange) is list and type(yRange) is list:
        xVec = np.linspace(xRange[0],xRange[1],int(np.sqrt(nPoints)))
        yVec = np.linspace(yRange[0],yRange[1],int(np.sqrt(nPoints)))
    else:
        xVec = xRange
        yVec = yRange
        
    xx,yy = np.meshgrid(xVec,yVec)
    xx,yy = xx.flatten(),yy.flatten()

    return pd.DataFrame(dict(zip(labels,[xx,yy])))
##############################################################


##############################################################
def polar_grid(r_sig,theta_sig,emitt = None):
    _coord  = generate_coordGrid(r_sig,theta_sig,labels = ['r_sig','theta_sig'])

    _coord.insert(2,'x_sig',_coord['r_sig']*np.cos(_coord['theta_sig']))
    _coord.insert(3,'y_sig',_coord['r_sig']*np.sin(_coord['theta_sig']))

    if emitt is not None:
        if isinstance(emitt, (int, float)):
            emitt = [emitt,emitt]

        _coord.insert(4,'x_n',_coord['x_sig']*np.sqrt(emitt[0]))
        _coord.insert(5,'y_n',_coord['y_sig']*np.sqrt(emitt[1]))
        _coord.insert(6,'J_x',(_coord['x_sig']**2)*emitt[0]/2)
        _coord.insert(7,'J_y',(_coord['y_sig']**2)*emitt[1]/2)

    return _coord
##############################################################