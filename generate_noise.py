"""

Library to create realistic randomly generated seismic noise from a background distribution
For cases where a single noise model is to be used, include something like:
peterson_NLNM.asc
which has two columns for:
period db

"""

#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import numpy as np

from numpy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import interp1d

def load_noise_model(fname):
    """
    Loads a noise model given in ascii format with two columns
    period    amplitude
    Returns frequency, amplitude, and a normalized 1d lookup function
    """
    f = open(fname,'r')
    periods = []
    amps = []
    for line in f.readlines():
        l = line.split()
        periods.append(float(l[0]))
        amps.append(float(l[1]))

    a = amps.copy()
    a += -np.min(a)
    a /= np.max(np.abs(a),axis=0)
    amp_func = interp1d(1.0/np.asarray(periods), a, fill_value='extrapolate')

    return [np.asarray(periods), np.asarray(amps), amp_func]


def add_earth_noise(trace, dt, scale, noise_model, phase_seed=None, amp_seed = None, phase_factor=1, amp_factor=1):
    """
    Adds frequency domain noise to a signal designed to match with an earth noise model
    amplitude is set by scale
    phase is set by random (pass phase_seed=None to randomize seed)

    dt is 1/sample rate as used in numpy's real fft function

    scale is a factor used to normalize between the noise model and the trace data. At scale=1 the noise amplitude
    will match the signal amplitude. At scale = 0 the noise amplitude will be 0.

    noise_model should be a 3 element list. The first element is an array of periods where the noise model is defined
    and the second is an array of amplitudes at those periods. The third element is a 1D interpolation function

    phase_factor is a perturbation on the randomization of the phase. Smaller values make the phase less perturbed.
    amp_factor is a perturbation on the randomization of the phase. Smaller values make the amplitude less perturbed
    returns: trace plus realistic earth noise
    """
    tr_spec = rfft(trace)
    tr_freqs = rfftfreq(len(trace), d=dt)
    tr_spec_amp = np.abs(tr_spec)
    tr_spec_phase = np.angle(tr_spec)

    trace_amp_scale = np.max(tr_spec_amp) - np.min(tr_spec_amp)

    # This term scales the noise depending on user value and reference trace
    noise_scale = trace_amp_scale * scale

    noise_spec_function = noise_model[2]

    rng = np.random.default_rng(amp_seed)
    noise_spec_r = (rng.random(len(tr_freqs)) - 0.5) * amp_factor
    noise_spec = noise_spec_function(tr_freqs) * noise_scale * noise_spec_r 
    noisy_tr_amp = tr_spec_amp + noise_spec
    rng = np.random.default_rng(phase_seed)

    noise_phase = 2*np.pi * rng.random(len(tr_spec_phase)) * phase_factor
    tr_phase_pos = tr_spec_phase + np.pi

    out_phase = (tr_phase_pos + noise_phase) % (2*np.pi) - np.pi

    noisy_spec = noisy_tr_amp * np.cos(out_phase) + 1j * noisy_tr_amp * np.sin(out_phase)

    return irfft(noisy_spec)

def dropout_trace(trace, dt, amplitude_scale):
    """
    Simulates a channel dropout by setting a flat value for the amplitude spectra
    input array trace with time sampling dt
    amplitude scale is the output scale relative to the peak amplitude spectrum of trace

    returns: new trace with constant amplitude spectrum
    """
    tr_spec = rfft(trace)
    tr_freqs = rfftfreq(len(trace), d=dt)
    tr_spec_amp = np.abs(tr_spec)
    tr_spec_phase = np.angle(tr_spec)

    dropout_amp = amplitude_scale * np.max(tr_spec_amp) * np.ones((len(tr_freqs),))
    dropout_phase = np.zeros((len(tr_freqs),))

    noisy_spec = dropout_amp * np.cos(dropout_phase) + 1j * dropout_amp * np.sin(dropout_phase)

    return irfft(noisy_spec)


def add_constant_background_noise(trace, dt, scale, noise_model, initial_phase, phase_seed=None, amp_seed = None,
                                  phase_factor=0.01, amp_factor=0.01):
    """
    as add_earth_noise, but designed to be applied over the entire array

    Most notable is the initial_phase input. Where add_earth_noise randomizes the noise phase, this initial phase
    term keeps the phase constant from trace to trace.

    scale sets the scaling between the noise amplitude spectrum and trace amplitude spectrum
    phase_factor and amp_factor are scalings applied on the randomization elements to keep them within
    reasonable bounds.

    """
    tr_spec = rfft(trace)
    tr_freqs = rfftfreq(len(trace), d=dt)
    tr_spec_amp = np.abs(tr_spec)
    tr_spec_phase = np.angle(tr_spec)

    trace_amp_scale = np.max(tr_spec_amp) - np.min(tr_spec_amp)

    # This term scales the noise depending on user value and reference trace
    noise_scale = trace_amp_scale * scale

    noise_spec_function = noise_model[2]

    rng = np.random.default_rng(amp_seed)
    noise_spec_r = (rng.random(len(tr_freqs)) - 0.5) * amp_factor
    noise_spec = noise_spec_function(tr_freqs) * noise_scale * noise_spec_r # Add randomization here

    noisy_tr_amp = tr_spec_amp + noise_spec
    rng = np.random.default_rng(phase_seed)

    # Add scaled noise to the initial phase input
    # Takes the initial phase input, adds random noise scaled by phase_factor, and then makes positive for
    # superposition
    # first, make the initial phase from 0 to 2 pi
    initial_phase_pos = (initial_phase + np.pi) % (2 * np.pi)
    # now definite a perturbation
    perturb_phase = rng.random(len(initial_phase))*phase_factor
    # now superimpose the initial and perturbation
    noise_phase = (initial_phase_pos + perturb_phase - np.pi) % (2 * np.pi)

    tr_phase_pos = tr_spec_phase + np.pi

    # superimpose the phase
    out_phase = (tr_phase_pos + noise_phase) % (2*np.pi) - np.pi

    # convert to real and imaginary for inverse fft
    noisy_spec = tr_spec + (noise_spec * np.cos(noise_phase) + 1j * noise_spec * np.sin(noise_phase))
    return irfft(noisy_spec)
