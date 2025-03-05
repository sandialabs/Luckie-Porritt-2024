###
# Utilities to compute source time functions as per CUDA2DSeis (internal cpp implementations)
###

#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import numpy as np
from obspy import Trace


def cast_to_trace(dt: float, wf: np.ndarray) -> Trace:
    """
    Given a time sampling value, dt and a time series, wf, returns a simple obspy Trace object.
    """
    tr = Trace()
    tr.data = wf
    tr.stats.delta = dt
    return tr

def trapozoid(dt, trap1, trap2, trap3):
    t1 = int(trap1//dt)
    t2 = int(trap2//dt)
    t3 = int(trap3//dt)
        
    t2 += t1
    lsrc = t2 + t3
    src = np.zeros((lsrc+1,))
    amp = (trap2 + 0.5 * (trap1 + trap3)) / dt

    jdx = 0
    for i in range(t1):
        src[jdx] = i / (t1 * amp)
        jdx += 1
    for i in range(t1, t2):
        src[jdx] = 1.0 / amp
        jdx += 1
    for i in range(t3, 0, -1):
        src[jdx] = i / (t3*amp)
        jdx += 1
    
    return lsrc, src
    
def gaussian(dt, alpha):
    aint = int(np.abs((int(alpha * 3.0))))
    lsrc = int(2 * aint + 1)
    src = np.zeros((lsrc+1, ))
    if alpha >= 0.0:
        for i in range(-aint, aint+1):
            src[i+aint] = i * np.exp(-i * i / (alpha * alpha))
    else:
        tmpsum = 0.0
        for i in range(-aint, aint+1):
            src[i+aint] = np.exp(-i * i / (alpha * alpha))
            tmpsum += src[i+aint]
        src /= tmpsum
    return lsrc, src
    
def ricker_wavelet(dt, npts, sigma=1):
    """
    defines a ricker wavelet in the time domain for the time array defined on dt and npts
    """
    t = np.arange(-npts*dt//2, npts*dt//2, dt)
    coeff1 = 2.0 / np.sqrt(3*sigma) * np.pi**(0.25)
    coeff2 = (1-(t/sigma)**2)
    coeff3 = np.exp(-t**2/(2*sigma**2))
    return coeff1 * coeff2 * coeff3

