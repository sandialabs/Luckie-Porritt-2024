# Set of functions for 1D and 2D beamforming and semblance analysis

#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import multiprocessing as mp
import numpy as np
import obspy
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def run_beamforming_1d_starmap(stream, reference = 0, duration = 0.1, step = 0.05, tmax = 2, smin = -0.005, smax = 0.005, s_step = 0.0001, ncpu=4):
    timewindows = []
    tstart = 0
    totalDuration = stream[reference].stats.endtime - stream[reference].stats.starttime
    if totalDuration < tmax:
        tmax = totalDuration
    tend = tstart + duration
    timewindows.append([tstart, tend])
    while tend < tmax:
        tstart += step
        tend += step
        timewindows.append([tstart, tend])

    Srvals = np.arange(smin, smax, s_step)

    if ncpu > mp.cpu_count():
        ncpu = mp.cpu_count()
    if ncpu <= 0:
        ncpu = 1
    pool = mp.Pool(int(ncpu))

    results = []
    results = pool.starmap(sem_stack_time_domain_1d_starmap, [(stream, sx, timewindows, reference, False) for sx in Srvals])
    pool.close()
    E1 = np.asarray(results)
    times = []
    for itime, tvals in enumerate(timewindows):
        times.append((tvals[0]+tvals[1])/2)

    return np.asarray(times), Srvals, E1


def parafft(stream, nthreads=None):
    """
    Computes the FFT for each trace in the stream.
    Adds stream[i].spectrum as complex valued vector
    Wrapper is primarily to split the task up to operate in parallel

    Function is done in place so output stream has the new spectrum appended
    Also returns the array of frequencies in the second output

    If nthreads is left as the default "none", then we will set it to 1
    If nthreads is greater than the cpu count, it will be set to ncpu - 1
    """
    # Just quickly get the array of frequencies. Assumes constant sampling rate and number of points
    freqs = np.fft.rfftfreq(stream[0].stats.npts, d=stream[0].stats.delta)

    # Get a parallel pool
    if nthreads >= mp.cpu_count():
        nthreads = int(mp.cpu_count()-1)
        print("Warning, requested too many threads. Reducing to ncpu - 1")
    if nthreads == None:
        nthreads = 1 # default
    pool = mp.Pool(int(round(nthreads)))

    results = []
    results = pool.starmap(fftsub, [(stream, idx, False) for idx in range(len(stream))])
    pool.close()
    for ir, rr in enumerate(results):
        idx = rr[0]
        spect = rr[1]
        stream[idx].spectrum = spect
    return stream, freqs

def run_frequency_domain_beamforming_starmap(stream, slownesses, frequencyarray, r0, nthreads=None):
    """
    Runs the stacksub in parallel over slownesses defined as a list of [slowness_x, slowness_y]

    """
    # Get a parallel pool
    if nthreads >= mp.cpu_count():
        nthreads = int(mp.cpu_count()-1)
        print("Warning, requested too many threads. Reducing to ncpu - 1")
    if nthreads == None:
        nthreads = 1 # default
    pool = mp.Pool(int(round(nthreads)))

    results = []
    results = pool.starmap(stacksub, [(stream, slowness, frequencyarray, r0) for slowness in slownesses])
    pool.close()

    return results

def fftsub(stream, index, verbose):
    d = stream[index].data
    spect = np.fft.rfft(d)
    if verbose:
        print("Done with {}".format(index))
    return [index, spect]

# Sampling problem
def setup_slowness_regular_grid(pxmin, pxmax, pxstep, pymin, pymax, pystep):
    px = np.arange(pxmin, pxmax+pxstep, pxstep)
    py = np.arange(pymin, pymax+pystep, pystep)
    p = []
    for ix, xtmp in enumerate(px):
        for iy, ytmp in enumerate(py):
            p.append([xtmp, ytmp])
    return p

def setup_slowness_azimuthal_grid(n_theta, n_slow, theta_min = -180, theta_max = 180, p_min = 0.1, p_max = 5.0, theta_deg = True):
    degToRad = np.pi/180
    if theta_deg:
        theta = np.linspace(theta_min, theta_max, n_theta)
        theta_rad = theta * degToRad
    else:
        theta_rad = np.linspace(theta_min, theta_max, n_theta)
        theta = theta_rad / degToRad
    pvals = np.linspace(p_min, p_max, n_slow)
    p = []
    for itheta, thetatmp in enumerate(theta_rad):
        for ip, ptmp in enumerate(pvals):
            xtmp = ptmp * np.cos(thetatmp)
            ytmp = ptmp * np.sin(thetatmp)
            p.append([xtmp, ytmp])
    return p, theta, pvals

def stacksub(stream, slowness, freqarray, r0):
    """
    # For a given slowness, stack the spectrums:
    #   for each station, compute k[f] . r
    #   sum_real[f] += T_real[f] * cos(k . r)
    #   sum_imag[f] += T_imag[f] * sin(k . r)
    # r0 is the reference location.
    #   It needs to be in the same coordinate frame as stream[i].stats.utmx and stream[i].stats.utmy
    """
    verbose = False

    # First, convert slowness to wavenumber
    nf = freqarray.size
    k = np.zeros((nf,2))
    sv = np.zeros(k.shape)
    for ix in range(nf):
        sv[ix, 0] = slowness[0]
        sv[ix, 1] = slowness[1]
    k[:,0] = 2.0 * np.pi * freqarray * sv[:,0]
    k[:,1] = 2.0 * np.pi * freqarray * sv[:,1]

    # Now, stack each station
    stack = np.zeros((nf,), dtype='complex')
    for itr, tr in enumerate(stream):
         # get r vector
        r = np.zeros((2,))
        r[0] = tr.stats.utmx - r0[0]
        r[1] = tr.stats.utmy - r0[1]
        # Now get the argument to the trig
        theta = np.zeros((nf,))
        for ifreq, freq in enumerate(freqarray):
            theta[ifreq] = np.dot(k[ifreq,:], r)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        stack.real += tr.spectrum.real * ctheta
        stack.imag += tr.spectrum.imag * stheta
    if verbose:
        print("Done with slowness: {:03.5f}, {:03.5f}".format(slowness[0], slowness[1]))
    return slowness, stack

def get_stream_median_location_utm(stream):
    x = np.zeros((len(stream),))
    y = np.zeros((len(stream),))
    for itr, tr in enumerate(stream):
        x[itr] = tr.stats.utmx
        y[itr] = tr.stats.utmy
    return [np.median(x), np.median(y)]

def get_stream_mean_location_utm(stream):
    x = np.zeros((len(stream),))
    y = np.zeros((len(stream),))
    for itr, tr in enumerate(stream):
        x[itr] = tr.stats.utmx
        y[itr] = tr.stats.utmy
    return [np.mean(x), np.mean(y)]

def get_stream_harmonic_mean_location_utm(stream):
    x = np.zeros((len(stream),))
    y = np.zeros((len(stream),))
    sumx = 0
    sumy = 0
    for itr, tr in enumerate(stream):
        x[itr] = tr.stats.utmx
        y[itr] = tr.stats.utmy
        if x[itr] != 0:
            sumx += 1/x[itr]
        if y[itr] != 0:
            sumy += 1/y[itr]
    return [len(stream)/sumx, len(stream)/sumy]

def disentangle_beamforming_stack(stack, frequencies):
    """
    Running the beamforming produces a list of lists, which gets really confusing
    Here we pull out what are needed for plotting or other manipulations (ie stacking over frequency range)

    Inputs:
    stack: This is the result from run_frequency_domain_beamforming_starmap
        At the top level, it is a list with n_slownesses elements
        each of these elements has two elements:
            1) [px, py]
            2) An array of n_frequencies of stack amplitudes

    frequencies: second output from parafft
        This is just the array of frequencies in the fft, generated by numpy based on sample rate and number
            of samples.

    The most common plot is px, py as independent x and y axes and colored by stack amplitude
    so outputs should basically be
    1) px vector
    2) py vector
    3) n_px by n_py by nfrequencies matrix of amplitudes
    """

    # First task is to count the unique slownesses in both px and py
    px = []
    py = []
    for ist, st in enumerate(stack):
        slowness_vector = st[0]
        if slowness_vector[0] not in px:
            px.append(slowness_vector[0])
        if slowness_vector[1] not in py:
            py.append(slowness_vector[1])
    n_px = len(px)
    n_py = len(py)
    n_f = len(frequencies)
    pxout = np.sort(np.asarray(px))
    pyout = np.sort(np.asarray(py))
    dpx = pxout[1] - pxout[0]
    dpy = pyout[1] - pyout[0]
    if dpx <= 0 or dpy <= 0:
        print("Error, dpx: {}; dpy: {}".format(dpx, dpy))
        return -1, -1, -1

    stack_array = np.zeros((n_px, n_py, n_f), dtype='complex')
    for ist, st in enumerate(stack):
        slowness_vector = st[0]
        amplitudes = st[1]
        ix = int(round((slowness_vector[0]-np.min(px))/ dpx))
        iy = int(round((slowness_vector[1]-np.min(py))/ dpy))
        if ix < 0 or iy < 0 or ix >= n_px or iy >= n_py:
            print("Error, ix: {} iy: {}".format(ix, iy))
        for ifreq, freq in enumerate(frequencies):
            stack_array[ix, iy, ifreq] += amplitudes[ifreq]

    return pxout, pyout, stack_array

def band_limited_beam_stack(stack_array, frequencies, fmin=0.1, fmax=2):
    """
    stacks the frequency dependent beamforming to within some band
    """
    (nx, ny, nf) = stack_array.shape
    stack_out = np.zeros((nx, ny), dtype='complex')

    for ifreq, freq in enumerate(frequencies):
        if freq >= fmin and freq <= fmax:
            stack_out += stack_array[:,:,ifreq]
    return stack_out

def sem_stack_time_domain_1d_starmap(stream, slowness, timewindows, iref, verbose):
    """
    Just packing some options from stack_time_domain_1d into defaults to allow multi-processing
    import multiprocess or multiprocessing as mp then:

    define data stream, Srvals, timewindows, and iref before
    """

    tr_ref = stream[iref].copy()
    stack = np.zeros((len(tr_ref.data),))
    for itr, tr in enumerate(stream):
        dx = (tr.stats.utmx - tr_ref.stats.utmx)
        dy = (tr.stats.utmy - tr_ref.stats.utmy)
        azi = np.arctan(dx/(dy+np.finfo(float).eps))
        pt = rotate_pt(dx, dy, -1*(90-azi))
        dt = pt[0] * slowness

        #nroll = np.int(np.floor(dt/tr_ref.stats.delta + 0.5))
        #shifted = np.roll(tr.data, nroll)
        #shifted[0:nroll] = 0.0
        shifted = np.zeros((len(tr.times()), ))
        for itime, tmptime in enumerate(tr.times()):
            tslant = tmptime - dt
            tslantidx = int(np.floor(tslant / tr.stats.delta+0.5))
            if tslantidx >= 0 and tslantidx < len(tr.times()):
                shifted[itime] =  tr.data[tslantidx]
        for idx in range(len(stack)):
            stack[idx] += shifted[idx]

    E = np.zeros((len(timewindows),))
    for itime, TimeWindow in enumerate(timewindows):
        Tmin = TimeWindow[0]
        Tmax = TimeWindow[1]

        # To get a function of time, we just need to adjust these windows:
        ib = np.int(np.floor((Tmin / tr_ref.stats.delta+0.5)))
        ie = np.int(np.ceil((Tmax / tr_ref.stats.delta-0.5)))

        # hard coded to sum mode
        E[itime] = np.sum(np.abs(stack[ib:ie+1]*tr_ref.stats.delta))

    if verbose:
        print("Done with slowness: {:02.6f}".format(slowness))
    return E



def cwt_semblance(stream, pmin, pmax, nslowness, L, amin, amax, verbose, iref):
    """
    Function similar to complex semblance, but iterates over scales in the cwt
    returns a semblance 2D structure for each scale

    Requires in the stream:
        stats.utmx and stats.utmy for location information
        cwt and cwt_scales for the wavelet domain representation and associated scales
            These can be added by first running langstonWavelets.add_cwt_to_stream(stream)

    pmin, pmax must be in seconds per meter (s/m)!
    amin and amax refer to the min and max scales to consider. Values outside these limits will be set to 0.
    (for reducing computation cost by ignoring scales outside of interest)

    Designed to be iterable over reference trace iref

    Basically considers each cwt[scale, timevector] as narrow band filters
    """
    # CWT is a complex representation, so the Hilbert transform is not necessary
    #from scipy.signal import hilbert

    pvector = np.linspace(pmin, pmax, nslowness)
    timesVector = stream[iref].times()
    # This will be significantly faster if the cwt is subspaced in scale
    nscales = len(stream[iref].cwt_scales)
    sem = np.zeros((nscales, len(pvector), len(timesVector)))

    # Initial loop to get number of traces to use
    ntraces = 0
    traceIndices = []
    for itmp in range(iref-L, iref+L+1):
        if itmp >= 0 and itmp < len(stream):
            traceIndices.append(itmp)
            ntraces += 1

    # Initialize a bunch of hilbert transforms and distances
    distanceVector = np.zeros((ntraces,))
    dataVector = np.zeros((nscales, ntraces, len(timesVector)), dtype='complex')

    # Second loop to get hilbert transforms and distances
    for idx, itmp in enumerate(traceIndices):
        #hilbertVector[idx,:] = hilbert(stream[itmp])
        for ia, a in enumerate(stream[iref].cwt_scales):
            if a >= amin and a <= amax:
                dataVector[ia, idx, :] = stream[itmp].cwt[ia,:]
        # Gah, this would be way more efficient outside the function. oh well.
        dx = (stream[itmp].stats.utmx - stream[iref].stats.utmx)
        dy = (stream[itmp].stats.utmy - stream[iref].stats.utmy)
        azi = np.arctan(dx/(dy+np.finfo(float).eps))
        pt = rotate_pt(dx, dy, (90-azi))
        distanceVector[idx] = pt[0]

    # Note the paper uses 1/(2L+1), but for most cases, that will be ntraces
    # for cases near the edge, that will decline, so adjusting here
    normFactor = 1/ntraces

    # semblance(px, t) = normFactor * (sum[real] + sum(imag)) / sum(real + imag)
    # So we need three sums over traces
    for ia, a in enumerate(stream[iref].cwt_scales):
        #print("Working on scale: {} {}".format(ia, a))
        if a >= amin and a <= amax:
            for ip, px in enumerate(pvector):
                for itime, tmptime in enumerate(timesVector):
                    tmpsum1 = 0
                    tmpsum2 = 0
                    tmpsum3 = 0
                    for idx, itmp in enumerate(traceIndices):
                        # Get the time index that is:
                        # px * (xj - x0)
                        tslant = tmptime + px * (distanceVector[idx])
                        tslantidx = int(np.floor(tslant / stream[itmp].stats.delta+0.5))
                        if tslantidx >= 0 and tslantidx < len(timesVector):
                            tmpsum1 += np.real(dataVector[ia, idx, tslantidx])
                            tmpsum2 += np.imag(dataVector[ia, idx, tslantidx])
                            tmpsum3 += np.real(dataVector[ia, idx, tslantidx])**2 + np.imag(dataVector[ia, idx, tslantidx])**2
                    tmpsum4 = tmpsum1**2 + tmpsum2**2
                    sem[ia, ip, itime] = normFactor * (tmpsum4 / tmpsum3)
            if verbose:
                print("Done with scale: {} {}".format(ia, a))

    return sem



def complex_semblance(stream, pmin, pmax, nslowness, iref, L):
    """
    From Shi and Huo "Complex Semblance and Its Application"
    doi: 10.1007/s12583-018-0829-x
    implemented by Lior et al in their equation 2
    It incorporates slowness (pmin, pstep, pmax) to slant stack
    and the Hilbert transform as the imaginary component (although scipy's Hilbert
    function does just that, original in real and transform in imag)

    Centers on trace iref and uses a gauge length L
    Although, as per Lior et al, it centers on iref and goes L to the left and L to the right

    This should work with starmap for parallel over iref

    Requires trace.stats.utmx and trace.stats.utmy for distance measurement

    pmin, pmax must be in seconds per meter (s/m)!

    This version only outputs the semblance.
    To get the vector of slownesses use:
        numpy.linspace(pmin, pmax, nslowness)
    To get the time vector use:
        stream[iref].times()
    """
    from scipy.signal import hilbert

    pvector = np.linspace(pmin, pmax, nslowness)
    timesVector = stream[iref].times()
    sem = np.zeros((len(pvector), len(timesVector)))

    # Initial loop to get number of traces to use
    ntraces = 0
    traceIndices = []
    for itmp in range(iref-L, iref+L+1):
        if itmp >= 0 and itmp < len(stream):
            traceIndices.append(itmp)
            ntraces += 1

    # Initialize a bunch of hilbert transforms and distances
    distanceVector = np.zeros((ntraces,))
    hilbertVector = np.zeros((ntraces, len(timesVector)), dtype='complex')

    # Second loop to get hilbert transforms and distances
    for idx, itmp in enumerate(traceIndices):
        hilbertVector[idx,:] = hilbert(stream[itmp])
        # Gah, this would be way more efficient outside the function. oh well.
        dx = (stream[itmp].stats.utmx - stream[iref].stats.utmx)
        dy = (stream[itmp].stats.utmy - stream[iref].stats.utmy)
        azi = np.arctan(dx/(dy+np.finfo(float).eps))
        pt = rotate_pt(dx, dy, (90-azi))
        distanceVector[idx] = pt[0]

    # Note the paper uses 1/(2L+1), but for most cases, that will be ntraces
    # for cases near the edge, that will decline, so adjusting here
    normFactor = 1/ntraces

    # semblance(px, t) = normFactor * (sum[real] + sum(imag)) / sum(real + imag)
    # So we need three sums over traces
    for ip, px in enumerate(pvector):
        for itime, tmptime in enumerate(timesVector):
            tmpsum1 = 0
            tmpsum2 = 0
            tmpsum3 = 0
            for idx, itmp in enumerate(traceIndices):
                # Get the time index that is:
                # px * (xj - x0)
                tslant = tmptime + px * (distanceVector[idx])
                tslantidx = int(np.floor(tslant / stream[itmp].stats.delta+0.5))
                if tslantidx >= 0 and tslantidx < len(timesVector):
                    tmpsum1 += np.real(hilbertVector[idx, tslantidx])
                    tmpsum2 += np.imag(hilbertVector[idx, tslantidx])
                    tmpsum3 += np.real(hilbertVector[idx, tslantidx])**2 + np.imag(hilbertVector[idx, tslantidx])**2
            tmpsum4 = tmpsum1**2 + tmpsum2**2
            sem[ip, itime] = normFactor * (tmpsum4 / tmpsum3)

    return sem



def get_beam_peaks(beamInfo2D, neighborhood_size=7, threshold=0.0001, gaussian=None):
    """
    returns x,y coordinates of peaks in a 2D beaminfo space (ie the output from a stacking method)
    """


    if gaussian is not None:
        beaminfo=gaussian_filter(beamInfo2D, gaussian)
    else:
        beaminfo = beamInfo2D.copy()

    data = beaminfo.T.copy()

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(np.int(x_center))
        y_center = (dy.start + dy.stop - 1)/2
        y.append(np.int(y_center))
    return x,y


def plot_beam_peaks(timeArray, pvalues, beaminfo, xpeak, ypeak):
    fig = plt.figure(figsize=(15,5))
    plt.pcolormesh(timeArray, pvalues, beaminfo.T, shading='auto', vmax=0.005)
    plt.xlabel("Time [sec]")
    plt.ylabel("slowness [sec/km]")
    plt.colorbar(label="Summed amplitude")
    plt.scatter(xpeak, ypeak,c='red')
    plt.show()

def rotate_pt(x, y, theta):
    """
    Rotates point (x, y) through angle defined by theta.

    Note that this angle is theta, ie normal math world angle, not azimuth
    To get theta from azimuth, take 90-azimuth
    If you want to "undo" a rotate, ie rotate along an array back to the x-axis, use -(90-azimuth)

    """
    rotmat = np.array([[np.cos(np.radians(theta)), -1*np.sin(np.radians(theta))],
                       [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    pt = np.array([x, y])
    rot = np.matmul(rotmat, pt)
    return rot


def stack_time_domain_1d_starmap(stream, slowness, timewindows, iref, verbose):
    """
    Just packing some options from stack_time_domain_1d into defaults to allow multi-processing
    import multiprocess or multiprocessing as mp then:

    define data stream, Srvals, timewindows, and iref before
    """

    tr_ref = stream[iref].copy()
    stack = np.zeros((len(tr_ref.data),))
    for itr, tr in enumerate(stream):
        dx = (tr.stats.utmx - tr_ref.stats.utmx)
        dy = (tr.stats.utmy - tr_ref.stats.utmy)
        azi = np.arctan(dx/(dy+np.finfo(float).eps))
        pt = rotate_pt(dx, dy, -1*(90-azi))
        dt = pt[0] * slowness

        nroll = np.int(np.floor(dt/tr_ref.stats.delta + 0.5))
        shifted = np.roll(tr.data, nroll)
        #shifted[0:nroll] = 0.0
        for idx in range(len(stack)):
            stack[idx] += shifted[idx]

    E = np.zeros((len(timewindows),))
    for itime, TimeWindow in enumerate(timewindows):
        Tmin = TimeWindow[0]
        Tmax = TimeWindow[1]

        # To get a function of time, we just need to adjust these windows:
        ib = np.int(np.floor((Tmin / tr_ref.stats.delta+0.5)))
        ie = np.int(np.ceil((Tmax / tr_ref.stats.delta-0.5)))

        # hard coded to sum mode
        E[itime] = np.sum(np.abs(stack[ib:ie+1]*tr_ref.stats.delta))

    if verbose:
        print("Done with slowness: {:02.6f}".format(slowness))
    return E




def stack_time_domain_1d(stream, slowness, timewindows, iref=0, amode='max'):
    """
    stack_time_domain_1d(stream, slowness, timewindows, iref=0, amod='max')
    modified from the regular stack_time_domain to only consider the radial direction
    """
    import matplotlib.pyplot as plt

    tr_ref = stream[iref].copy()
    stack = np.zeros((len(tr_ref.data),))
    for itr, tr in enumerate(stream):
        dx = (tr.stats.utmx - tr_ref.stats.utmx)
        dy = (tr.stats.utmy - tr_ref.stats.utmy)
        azi = np.arctan(dx/(dy+np.finfo(float).eps))
        theta = -1 * (90 - azi)
        pt = rotate_pt(dx, dy, theta)
        dr = pt[0]
        dt = 1.0 * (dr * slowness)

        nroll = np.int(np.floor(dt/tr_ref.stats.delta + 0.5))
        shifted = np.roll(tr.data, nroll)
        for idx in range(len(stack)):
            stack[idx] += shifted[idx]

    E = np.zeros((len(timewindows),))
    for itime, TimeWindow in enumerate(timewindows):
        Tmin = TimeWindow[0]
        Tmax = TimeWindow[1]
        if Tmax < Tmin:
            tmpmin = Tmax
            tmpmax = Tmin
            Tmin = tmpmin
            Tmax = tmpmax
        if Tmax == Tmin:
            print("Error, Tmax == Tmin at {}".format(itime))
            return -9002

        # To get a function of time, we just need to adjust these windows:
        ib = np.int(np.floor((Tmin / tr_ref.stats.delta+0.5)))
        ie = np.int(np.ceil((Tmax / tr_ref.stats.delta-0.5)))

        if amode == "sum":
            E[itime] = np.sum(np.abs(stack[ib:ie+1]*tr_ref.stats.delta))
        else:
            E[itime] = np.max(stack[ib:ie+1])

    return E


def stack_time_domain_starmap(stream, sx, sy, timewindows, iref):
    """
    stack_time_domain_starmap(stream, sx, sy, timewindows, iref)
    As per stack_time_domain() but changes the options a bit for use with multiprocessing

    """
    import matplotlib.pyplot as plt

    tr_ref = stream[iref].copy()
    stack = np.zeros((len(tr_ref.data),))
    for itr, tr in enumerate(stream):
        dx = (tr.stats.utmx - tr_ref.stats.utmx)
        dy = (tr.stats.utmy - tr_ref.stats.utmy)
        if hasattr(tr.stats,'correctionTime'):
            dt = -1.0 * (dx * sx + dy * sy) + tr.stats.correctionTime # not sure if this should be a 1 or a -1?
        else:
            dt = -1.0 * (dx * sx + dy * sy)

        nroll = np.int(np.floor(dt/tr_ref.stats.delta + 0.5))
        shifted = np.roll(tr.data, nroll)
        for idx in range(len(stack)):
            stack[idx] += shifted[idx]

    E = np.zeros((len(timewindows),))
    for itime, TimeWindow in enumerate(timewindows):
        Tmin = TimeWindow[0]
        Tmax = TimeWindow[1]
        if Tmax < Tmin:
            tmpmin = Tmax
            tmpmax = Tmin
            Tmin = tmpmin
            Tmax = tmpmax
        if Tmax == Tmin:
            print("Error, Tmax == Tmin at {}".format(itime))
            return -9002

        # To get a function of time, we just need to adjust these windows:
        ib = np.int(np.floor((Tmin / tr_ref.stats.delta+0.5)))
        ie = np.int(np.ceil((Tmax / tr_ref.stats.delta-0.5)))

        amode = 'sum'

        if amode == "sum":
            E[itime] = np.sum(np.abs(stack[ib:ie+1]*tr_ref.stats.delta))
        else:
            E[itime] = np.max(stack[ib:ie+1])

    return E

def stack_time_domain(stream, sx, sy, timewindows, iref=0, amode="max"):
    """
    stack_time_domain(stream, sx, sy, timewindows, iref=0, amode="max")
    stacks in the time domain the data in stream, for reference station iref at slowness sx and sy
    outputs in E[itime] for each time window defined in timewindows
    amode="sum" will stack up the data in the time window otherwise we simply pick the peak

    """
    

    tr_ref = stream[iref].copy()
    stack = np.zeros((len(tr_ref.data),))
    for itr, tr in enumerate(stream):
        dx = (tr.stats.utmx - tr_ref.stats.utmx)
        dy = (tr.stats.utmy - tr_ref.stats.utmy)
        dt = -1.0 * (dx * sx + dy * sy)

        nroll = np.int(np.floor(dt/tr_ref.stats.delta + 0.5))
        shifted = np.roll(tr.data, nroll)
        for idx in range(len(stack)):
            stack[idx] += shifted[idx]

    E = np.zeros((len(timewindows),))
    for itime, TimeWindow in enumerate(timewindows):
        Tmin = TimeWindow[0]
        Tmax = TimeWindow[1]
        if Tmax < Tmin:
            tmpmin = Tmax
            tmpmax = Tmin
            Tmin = tmpmin
            Tmax = tmpmax
        if Tmax == Tmin:
            print("Error, Tmax == Tmin at {}".format(itime))
            return -9002

        # To get a function of time, we just need to adjust these windows:
        ib = np.int(np.floor((Tmin / tr_ref.stats.delta+0.5)))
        ie = np.int(np.ceil((Tmax / tr_ref.stats.delta-0.5)))

        if amode == "sum":
            E[itime] = np.sum(np.abs(stack[ib:ie+1]*tr_ref.stats.delta))
        else:
            E[itime] = np.max(stack[ib:ie+1])

    return E

def synth_forward_shift(timeArray, points, sourceFunction="ricker", Sx=0.0, Sy=0.0, sigma=1):
    """
    Creates a npts x ntimes ndarray with data for the sourceFunction (initially must be "ricker")
    Data has been shifted by Sx and Sy
    timeArray must be symmetric about 0

    """
    slowvec = np.array([Sx, Sy])
    a = np.dot(slowvec, points.T)

    min_t = np.min(timeArray)
    max_t = np.max(timeArray)
    dt = timeArray[1]-timeArray[0]

    timearray = np.arange(min_t, max_t, dt)

    if sourceFunction=="ricker":
        amparray = ricker_wavelet_td(timearray,sigma=sigma)
    else:
        amparray = ricker_wavelet_td(timearray,sigma=sigma)

    dataarray = np.zeros((len(points), len(amparray)))
    for idx in range(len(points)):
        tshift = a[idx]
        tmptime = timearray + tshift
        nptsshift = int(np.floor(tshift / dt+0.5))
        dataarray[idx,:] = np.roll(amparray, nptsshift)

    return dataarray

def ricker_wavelet_td(t, sigma=1):
    """
    defines a ricker wavelet in the time domain for the time array t
    """
    import numpy as np
    coeff1 = 2.0 / np.sqrt(3*sigma) * np.pi**(0.25)
    coeff2 = (1-(t/sigma)**2)
    coeff3 = np.exp(-t**2/(2*sigma**2))
    return coeff1 * coeff2 * coeff3

def delta_fn_td(t, peak=1):
    """
    defines a delta function (spike) in the center of the time window
    """
    import numpy as np
    out = np.zeros((len(t),))
    icenter = int(np.floor(0.5+np.mean([t[0], t[-1]])))
    out[icenter] = peak
    return out

def synth_stream_array(stream, srcx=0, srcy=0):
    """
    uses stream[i].stats.utmx and stream[i].stats.utmy to create a set of points for synthetic beam forming
    offsets x and y by srcx and srcy respectively
    """
    import numpy as np
    n = len(stream)
    outarray = np.zeros((n,2))
    for itr, tr in enumerate(stream):
        outarray[itr, 0] = tr.stats.utmx - srcx
        outarray[itr, 1] = tr.stats.utmy - srcy
    return outarray


def synth_line_array(n=5, azimuth=0, min_r = -5, max_r = 5, x0=0, y0=0):
    """
    creates a matrix with n stations at an azimuth, azimuth with distances from min_r to max_r
    """
    import numpy as np

    outarray = np.zeros((n,2))
    r = np.linspace(min_r, max_r, n)
    cx = np.cos(np.radians(90-azimuth))
    sy = np.sin(np.radians(90-azimuth))

    outarray[:,0] = r * cx + x0
    outarray[:,1] = r * sy + y0

    return outarray

def synth_circle_array(n=5, r=1, x0=0, y0=0):
    """
    creates a matrix with n stations around a circle of radius r
    """
    import numpy as np

    outarray = np.zeros((n,2))
    dangle = 360/n
    for idx in range(n):
        tmpx = r * np.cos(np.radians(dangle*idx))+x0
        tmpy = r * np.sin(np.radians(dangle*idx))+y0
        outarray[idx,0] = tmpx
        outarray[idx,1] = tmpy

    return outarray

def synth_random_array(n=5, sigmax=1, sigmay=1, meanx=0, meany=0):
    '''
    defines a set of random locations with gaussian distributions in x and y
    '''
    import numpy as np
    from numpy.random import normal

    outarray = np.zeros((n,2))
    x = normal(loc=meanx, scale=sigmax, size=n)
    y = normal(loc=meany, scale=sigmay, size=n)
    outarray[:,0] = x
    outarray[:,1] = y
    return outarray

def utm_plot_stream(stream, iref=0, grid=True, xlabel = "utmx [meters]", ylabel="utmy [meters]", title="Array Geometry", figsize=(7,7)):
    """
    uses tr.stats.utmx and tr.stats.utmy for trace "tr" in stream to plot a map view of the stations
    plots the reference trace, identified as iref, as a second color to identify it
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=figsize)
    utmx = np.zeros((len(stream),))
    utmy = np.zeros((len(stream),))

    for itr, tr in enumerate(stream):
        utmx[itr] = tr.stats.utmx
        utmy[itr] = tr.stats.utmy

    plt.scatter(utmx, utmy)
    plt.scatter(stream[iref].stats.utmx, stream[iref].stats.utmy)
    if grid:
        plt.grid()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.show()

def check_peak(E):
    """
    returns R=Pmax/Ptotal
    as per Langston phased array equation 13
    This is a rough quality factor to see if a peak stands out significantly against the background
    """
    import numpy as np

    ind = np.argmax(E)
    Etmp = np.ravel(E)
    Pmax = Etmp[ind]
    Ptotal = np.sum(E)
    return Pmax/Ptotal



def find_peak_baz_and_slowness(x, y, E):
    """
    Just picks the peak (argmax) from the 2D array E and returns the:
        azimuth, magnitude of the slowness, Sx, and Sy
    needs the Sx and Sy vectors that were used to generate E
    """
    import numpy as np

    ind = np.unravel_index(np.argmax(E, axis=None), E.shape)
    sx = x[ind[0]]
    sy = y[ind[1]]

    azi = 180/np.pi * np.arctan2(sx, sy) # Note that we switch x and y from the usual to get azimuth
    mag = np.sqrt(sx**2 + sy**2)
    return azi, mag, sx, sy

def add_spectrum_to_stream(stream, stype='fft', wtype='morlet',nv=16):
    """
    Adds either a complex valued fft or wavelet to the stream for each trace
    """
    import numpy as np
    import numpy.fft as FFT
    import langstonWavelets as lw

    streamout = stream.copy()
    for itr, tr in enumerate(streamout):
        if stype == 'fft':
            spect = FFT.rfft(tr.data)
            freq = FFT.rfftfreq(tr.stats.npts, d=1./tr.stats.sampling_rate)
        elif stype == 'cwt':
            spect, period, opt = lw.cwt_fw(tr.data, wtype, nv, tr.stats.delta)
            freq = 1.0/period
        tr.spect = spect
        tr.freq = freq
    return streamout

def stack_frequency_domain(stream, sx, sy, Twindows, iref=0, stype='fft', wtype='morlet', nv=16):
    """
    Follows the stack function for beamforming after spectrum has been computed and stored in the stream
    searches for slowness between sx and sy and amplitudes in time from Tmin to Tmax
    Will work for two modes: either a spectrum in the Fourier domain or the wavelet domain, but intially will be made for Fourier

    Twindows is a list of two element [tmin, tmax] sublists that are expanded to get E(t)

    """
    import numpy as np
    import numpy.fft as FFT

    stackarray = np.zeros((stream[iref].stats.npts,),dtype='complex')
    tr_ref = stream[iref].copy()

    if stype == 'fft':
        for itr, tr in enumerate(stream):
            dx = (tr.stats.utmx - tr_ref.stats.utmx) / 1000
            dy = (tr.stats.utmy - tr_ref.stats.utmy) / 1000
            dt = -1.0 * (sx * dx + sy * dy)

            for ifreq, freq in enumerate(tr_ref.freq):
                arg = -2 * np.pi * freq * dt
                cs = np.cos(arg)
                sn = np.sin(arg)
                rr = tr.spect[ifreq].real * cs - tr.spect[ifreq].imag * sn
                ii = tr.spect[ifreq].real * sn + tr.spect[ifreq].imag * cs
                stackarray.real += rr
                stackarray.imag += ii

        # Return the stack to the time domain
        corr_stack = FFT.irfft(stackarray)  # 90% certain this is the slow part

        # Get the absolute value/envelope
        corr_stack = np.abs(corr_stack)

        E = np.zeros((len(Twindows),))
        for itime, TimeWindow in enumerate(Twindows):
            Tmin = TimeWindow[0]
            Tmax = TimeWindow[1]
            if Tmax < Tmin:
                tmpmin = Tmax
                tmpmax = Tmin
                Tmin = tmpmin
                Tmax = tmpmax
            if Tmax == Tmin:
                print("Error, Tmax == Tmin at {}".format(itime))
                return -9002

            # To get a function of time, we just need to adjust these windows:
            ib = np.int(np.floor((Tmin / tr_ref.stats.delta+0.5)))
            ie = np.int(np.ceil((Tmax / tr_ref.stats.delta-0.5)))

            E[itime] = np.sum(corr_stack[ib:ie+1]*tr_ref.stats.delta)
            #for idx in range(ib, ie+1):
            #    E[itime] += corr_stack[idx]*tr_ref.stats.delta

        return E
    elif stype == 'cwt':
        print("in progress")
        return -9001

    # For cwt, can we use interpolation to get t+p_x * x + p_y * y?

def stack_frequency_domain_starmap_fft(stream, sx, sy, Twindows, iref):
    """
    Follows the stack function for beamforming after spectrum has been computed and stored in the stream
    searches for slowness between sx and sy and amplitudes in time from Tmin to Tmax
    Will work for two modes: either a spectrum in the Fourier domain or the wavelet domain, but intially will be made for Fourier

    Twindows is a list of two element [tmin, tmax] sublists that are expanded to get E(t)

    """
    import numpy as np
    import numpy.fft as FFT

    stackarray = np.zeros((stream[iref].stats.npts,),dtype='complex')
    tr_ref = stream[iref].copy()


    for itr, tr in enumerate(stream):
        dx = (tr.stats.utmx - tr_ref.stats.utmx) / 1000
        dy = (tr.stats.utmy - tr_ref.stats.utmy) / 1000
        dt = -1.0 * (sx * dx + sy * dy)

        for ifreq, freq in enumerate(tr_ref.freq):
            arg = -2 * np.pi * freq * dt
            cs = np.cos(arg)
            sn = np.sin(arg)
            rr = tr.spect[ifreq].real * cs - tr.spect[ifreq].imag * sn
            ii = tr.spect[ifreq].real * sn + tr.spect[ifreq].imag * cs
            stackarray.real += rr
            stackarray.imag += ii

    # Return the stack to the time domain
    corr_stack = FFT.irfft(stackarray)  # 90% certain this is the slow part

    # Get the absolute value/envelope
    corr_stack = np.abs(corr_stack)


    E = np.zeros((len(Twindows),))
    for itime, TimeWindow in enumerate(Twindows):
        Tmin = TimeWindow[0]
        Tmax = TimeWindow[1]
        if Tmax < Tmin:
            tmpmin = Tmax
            tmpmax = Tmin
            Tmin = tmpmin
            Tmax = tmpmax
        if Tmax == Tmin:
            print("Error, Tmax == Tmin at {}".format(itime))
            return -9002


        # To get a function of time, we just need to adjust these windows:
        ib = np.int(np.floor((Tmin / tr_ref.stats.delta+0.5)))
        ie = np.int(np.ceil((Tmax / tr_ref.stats.delta-0.5)))

        E[itime] = np.sum(corr_stack[ib:ie+1]*tr_ref.stats.delta)
        #for idx in range(ib, ie+1):
        #    E[itime] += corr_stack[idx]*tr_ref.stats.delta

    return E

    # For cwt, can we use interpolation to get t+p_x * x + p_y * y?
