#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

# class file to define sources and receivers outside of the CUDA2DSimulator context.
# Shouldn't be necessary, but as a backup
from obspy import UTCDateTime
from math import log10
from geoUtils import distaz

class Receiver:

    def __init__(self, coords=(0.0, 0.0, 0.0), numSensors=1, sensordx=0, sensordz=0, stationName = "SYNTH", network="SY", location = "00"):
        self.setCoords(coords)
        self.directoryName = ""
        self.numSensors = numSensors
        self.sensordx = sensordx
        self.sensordz = sensordz
        self.kstnm = stationName
        self.knetwk = network
        self.location = location

    def setCoords(self, coords):
        self.__coords = coords
        self.lon = self.__coords[0]
        self.lat = self.__coords[1]
        self.elevation = self.__coords[2]

    def getCoords(self):
        return self.__coords

    def __str__(self):
        return "Receiver {}:({},{},{})".format(self.kstnm, self.lat, self.lon, self.elevation)

    def __repr__(self):
        return "Receiver(coords=({},{},{}),name={})".format(self.lat, self.lon, self.elevation, self.kstnm)


class Source:

    def __init__(self, coords=(0.0, 0.0, 0.0), name=None):
        self.setCoords(coords)
        self.setName(name)
        self.strike = None
        self.dip = None
        self.rake = None
        self.expToDcRatio = None
        self.M0 = None
        self.Mw = None
        self.gcmt = None
        self.parameters = {}
        self.rise = None
        self.duration = None
        self.dt = None

    def setCoords(self, coords):
        self.__coords = coords
        self.lon = self.__coords[0]
        self.lat = self.__coords[1]
        self.elevation = self.__coords[2]

    def setName(self, name):
        self.name = name

    def setOriginTime(self, originTime):
        self.originTime = UTCDateTime(originTime)

    def addDislocationComponent(self, strike, dip, rake):
        self.strike = strike
        self.dip = dip
        self.rake = rake
        self.parameters['strike'] = self.strike
        self.parameters['dip'] = self.dip
        self.parameters['rake'] = self.rake

    def addVolumetricComponent(self, amplitudeRatio=1.0):
        self.expToDcRatio = amplitudeRatio
        
    def addMagnitude(self, M0 = None, Mw = None):
        import numpy as np
        if M0 is not None:
            self.M0 = M0
            self.Mw = (2.0/3.0) * log10(M0) - 10.7
        else:
            self.Mw = Mw
            self.M0 = np.power((3.0/2.0) * (10.7+Mw),10)

    def setSourceTimeFunction(self, type='Gaussian', alphaType='default', **kwargs):
        if type == 'Gaussian':
            self.sourceType = 'Gaussian'
            if alphaType == 'seconds':
                # the alpha pulse width is in seconds. We want to change to alpha as used in the 2d code:
                dt = kwargs['dt']
                self.alpha =  float(kwargs['alpha']) / (4.0 * dt)
            else:
                self.alpha = kwargs['alpha']
            self.trap1 = ""
            self.trap2 = ""
            self.trap3 = ""
            self.parameters['srctime'] = 'g'
        if type == 'triangle':
            self.sourceType = 'triangle'
            self.alpha = ""
            self.trap1 = kwargs['trap1']
            self.trap2 = kwargs['trap2']
            self.trap3 = kwargs['trap3']
            self.parameters['srctime'] = 't'
        self.parameters['alpha'] = self.alpha
        self.parameters['trap1'] = self.trap1
        self.parameters['trap2'] = self.trap2
        self.parameters['trap3'] = self.trap3
        
    def setSourceDurationRise(self, duration=0, rise=0.5, dt=0.1):
        self.duration = duration
        self.rise = rise
        self.dt = dt
        
    def load_gcmt(self, event):
        """
        Use the obspy function read_events on a GCMT file.
        This function then casts that into an event object for conversion later
        """
        self.gcmt = event

    def __str__(self):
        return "Source {}:({},{},{}) strike={}, dip={}, rake={}".format(self.name, self.lat, self.lon, self.elevation,
                                                                        self.strike, self.dip, self.rake)

    def __repr__(self):
        return "Source(coords=({},{},{}),name={})".format(self.lat, self.lon, self.elevation, self.name)

    
def source_to_receiver(source: Source, receiver: Receiver) -> list:
    dist, xdeg, azi = distaz(source.lat, source.lon, receiver.lat, receiver.lon)
    return [dist, xdeg, azi]