# PYFKSimulator.py
# Wrapper code to run pyfk - a 1D pure python frequency wavenumber synthetic seismogram tool
# While the base pyfk is fairly easy to run, this wrapper is designed so the functionality is as close as possible to CUDA2DSimulator.py

#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import numpy as np
import obspy, os, pickle
from pyfk.config.config import Config, SeisModel, SourceModel # Main PyFK classes for configuration, structure, and source, respectively
from pyfk.gf.gf import calculate_gf # Main function to calculate green's function for a source-receiver
from pyfk.tests.taup.test_taup import TestFunctionTaup # allows quick build of PREM model
from pyfk.sync.sync import calculate_sync, generate_source_time_function # functions that are convolved with the GF
from pyfk.utils.error_message import PyfkError # Error handling

# Based on functionality of CUDA2DSimulator (for multiprocessing)
from sys import platform
if platform == 'darwin':
    import multiprocess as mp
else:
    import multiprocessing as mp
    
from stfUtils import trapozoid, gaussian

# Bring in the source and receiver classes
from SourceAndReceivers import Receiver, Source

# Little utility for computing distance and azimuth between source and receiver objects
from SourceAndReceivers import source_to_receiver

# Access the abstract 1D model class
from SimulatorModels import Model1D

# Will need to get distance at least
from geoUtils import distaz

# This gets used in the tomography model class
from scipy.io import netcdf as nc

simDir = "Simulations"

class executioner():
    def __init__(self, paramSet):
        self.paramSet = paramSet
        self.process = Process()
        return

    def execute(self, sim):
        if sim is not None:
            print("Hangman {} initiating process".format(self.paramSet))
            self.process = Process(target=sim.execute, args=(self.paramSet,))
            self.process.start()
        return

    def stage(self, sim):
        if sim is not None:
            print("Hangman {} staging process".format(self.paramSet))
            self.process = Process(target=sim.stage, args=(self.paramSet,))
            self.process.start()        
        return
    
    
class simulationCollection(list):
    def __init__(self, iterable):
        super().__init__(item for item in iterable)
        
    def __setitem__(self, index, item):
        super().__setitem__(index, item)

    def insert(self, index, item):
        super().insert(index, item)
    
    def execute(self):
        for item in self:
            item.execute()
            
    def execute_mp(self, nthreads=1, verbose=True):
        print("Executing simulations on {:0.0f} threads.".format(nthreads))
        with mp.Pool(nthreads) as pool:
            result = pool.map(self._execute_sim, [(idx) for idx in range(len(self))])
        self.result = result
        self._repack_results()
        
    def _repack_results(self):
        for idx in range(len(self.result)):
            self.__setitem__(self.result[idx][0], self.result[idx][1])
       
    def _execute_sim(self, index):
        self[index].execute()
        return [index, self[index]]
        
    def view_simulation(self, index=0):
        print(self[index])
        
    def view_simulations(self):
        for item in self:
            print(item)
        
    def append(self, item):
        super().append(item)

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(item for item in other)

class PYFKModel():
    """
    Wrapper to pyfk.config.config SeisModel to operate as CUDA2DSimulator model
    """
    def __init__(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        self.model = SeisModel(model=model_data)  
        return
        
    # These three setters are using the three models that come with pyfk as .nd files
    def set_prem(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        self.model_data = model_data
        self.model = SeisModel(model=model_data)
        return
        
    def set_ak135(self):
        model_data = TestFunctionTaup.gen_test_model("ak135f_no_mud")
        self.model_data = model_data
        self.model = SeisModel(model=model_data)
        return
    
    def set_1066a(self):
        model_data = TestFunctionTaup.gen_test_model("1066a")
        self.model_data = model_data
        self.model = SeisModel(model=model_data)
        return
    
    def convert_Model1D(self, model: Model1D):
        nlayers = len(model.layer_thickness)
        model_data = np.zeros((nlayers, 6)) 
        for idx in range(nlayers):
            model_data[idx, 0] = model.layer_thickness[idx]
            model_data[idx, 1] = model.vs[idx]
            model_data[idx, 2] = model.vp[idx]
            model_data[idx, 3] = model.density[idx]
            model_data[idx, 4] = model.Qs[idx]
            model_data[idx, 5] = model.Qp[idx]
        self.model_data = model_data
        self.model = SeisModel(model=model_data)
        return
    
    # Need to add:
    #   Crust1
    #   addPerturbation
    #   


class PYFKSimulation():
    def __init__(self, simDir, simID = "00000000", numGPUs=0, simTypes = []):
        self.simTypes = simTypes
        self.simDir = simDir
        self.simID = simID
        os.makedirs(simDir, exist_ok = True)
        self.numGPUs = numGPUs # Doesn't actually do anything since this is a CPU simulation
        self.receiver = None
        self.source = None
        self.model = None
        self.parameters = []
        self.doDCSource = False
        self.doEXPSource = False
        self.doSFSource = False
        self.doGCMTSource = False
        self.sourceDC = None
        self.sourceEXP = None
        self.sourceSF = None
        self.sourceGCMT = None
        self.Completed = False
        self.greens_functions = []
        return
    
    def setSimulationParameters(self, npt = None, dt = None): # What PYFK calls a Config object
        # Use source and receivers to define receiver_distance
        self._cast_Receiver_to_distances()
        
        # convert source from Source class to SourceModel class
        self._cast_Source_to_SourceModel()
    
        if self.doDCSource:
            conf = Config(
                model = self.model,
                source = self.sourceDC,
                npt = npt,
                dt = dt,
                receiver_distance = self.distances)
            self.parameters.append(conf)
            
        if self.doEXPSource:
            conf = Config(
                model = self.model,
                source = self.sourceEXP,
                npt = npt,
                dt = dt,
                receiver_distance = self.distances)
            self.parameters.append(conf)
            
        if self.doGCMTSource:
            conf = Config(
                model = self.model,
                source = self.sourceGCMT,
                npt = npt,
                dt = dt,
                receiver_distance = self.distances)
            self.parameters.append(conf)
            
        if self.doSFSource:
            conf = Config(
                model = self.model,
                source = self.sourceSF,
                npt = npt,
                dt = dt,
                receiver_distance = self.distances)
            self.parameters.append(conf)
        return
            
    def setReceiver(self, receiver: Receiver):
        self.receiver = receiver
        return
        
    def setSource(self, source: Source):
        self.source = source
        if self.source.expToDcRatio is not None:
            self.simTypes.append("EXP")
        return
            
    def setModel(self, model: PYFKModel):
        self.model = model.model
        return
                
    # Helper internal functions for casting source and receiver info
    def _cast_Receiver_to_distances(self):
        nrec = self.receiver.numSensors
        assert self.source is not None, "Error, source location is required to define receiver offsets."
        dist0, gcarc, azi = distaz(self.source.lat, self.source.lon, self.receiver.lat, self.receiver.lon)
        self.distances = np.zeros((nrec,))
        for rec in range(nrec):
            self.distances[rec] = dist0 + rec * self.receiver.sensordx
        return
        
    def _cast_Source_to_SourceModel(self):
        # SourceModel takes sdep, srcType, and source_mechanism
        # Source has coords tuple, name, strike, dip, rake, expToDcRatio, M0, Mw, and parameters
        #   parameters is a dictionary 'strike', 'dip', and 'rake'
        # At setSourceTimeFunction, it adds sourceType, alpha, trap1, trap2, trap3
        #   Adds parameters to dictionary 'srctime', 'alpha', 'trap1', 'trap2', 'trap3'
        # coords order is longitude, latitude, elevation
        sdep = self.source.elevation
        self.doDCSource = False
        self.doEXPSource = False
        self.doSFSource = False
        self.doGCMTSource = False
        
        # Create source objects for double-couple, single force, or explosion
        if self.source.strike is not None:
            # Double-couple
            self.doDCSource = True
            if self.source.Mw is not None:
                mag = self.source.Mw
            elif self.source.M0 is not None:
                mag = (2.0/3.0) * log10(self.source.M0) - 10.7
            else:
                mag = 1.0
            src_mech = [mag, self.source.strike, self.source.dip, self.source.rake]
            self.sourceDC = SourceModel(sdep=sdep, srcType='dc', source_mechanism=src_mech)
            
        if self.source.gcmt is not None:
            self.doGCMTSource = True
            self.sourceGCMT = SourceModel(sdep=sdep)
            self.sourceGCMT.update_source_mechanism(self.source.gcmt)
            
        if self.source.expToDcRatio is not None:
            self.doEXPSource = True
            if self.source.M0 is not None:
                mag = self.source.M0 * self.source.expToDcRatio
            elif self.source.Mw is not None:
                mag = 10**((3.0/2.0) * (Mw + 10.7))
                mag *= self.source.expToDcRatio
            else:
                mag = 1.0 * self.source.expToDcRatio
            self.sourceEXP = SourceModel(sdep=sdep, srcType='ep', source_mechanism = [mag])
        return
            
    def execute(self, gpuID = None, verbose=False):
        if gpuID is not None:
            gpuID = None
        cwd = os.getcwd()
        if verbose:
            print("Executing Simulation in Directory: {}".format(self.simDir))
        if not os.path.exists(self.simDir):
            os.makedirs(self.simDir)
        os.chdir(self.simDir)
        greens_functions = []
        with open('PYFK_Config_{}.pkl'.format(self.simID), 'wb') as f:
            pickle.dump(self.parameters, f)
        for conf in self.parameters:
            gf = calculate_gf(conf)
            greens_functions.append(gf)
        self.greens_functions = greens_functions
        with open('PYFK_GF_{}.pkl'.format(self.simID), 'wb') as f:
            pickle.dump(self.greens_functions, f)
        self.Completed = True
        os.chdir(cwd)
        return
    
    def processData(self, outputStream = True, outputSAC = False, stf=None):
        """
        Convert the greens functions waveforms into 3 component seismograms from each simulation
        Thin wrapper to:
            calculate_sync(
                gf: Union[List[obspy.core.stream.Stream], obspy.core.stream.Stream],
                config: pyfk.config.config.Config,
                az: Union[float, int] = 0,
                source_time_function: Union[obspy.core.trace.Trace, NoneType] = None,
            ) -> List[obspy.core.stream.Stream]
        """
        self.displacements = []
        for idx, conf in enumerate(self.parameters):
            tmp_displacements = []
            for wf in self.greens_functions:
                [dist, gcarc, az] = source_to_receiver(self.source, self.receiver)
                displacement = calculate_sync(wf, conf, az, stf)
                tmp_displacements.append(displacement)
            self.displacements.append(tmp_displacements)
        if outputStream:
            cwd = os.getcwd()
            os.chdir(self.simDir)
            with open("PYFK_DISPLACEMENT_{}.pkl".format(self.simID), "wb") as f:
                pickle.dump(self.displacements, f)
            os.chdir(cwd)
        # Flags for saving output
            #if outputStream:
            #    fname = 
        
        return
        
        

    def __str__(self):
        mystr = "Simulation Configurations: \n\n"
        for idx, conf in enumerate(self.parameters):
            mystr += "{:01.0f}: {} \n\n".format(idx, conf)
        mystr += "Completed: {}\n\n".format(self.Completed)
        return mystr

    def __repr__(self):
        mystr = "Simulation Configurations: \n\n"
        for idx, conf in enumerate(self.parameters):
            mystr += "{:01.0f}: {} \n\n".format(idx, conf)
        mystr += "Completed: {}\n\n".format(self.Completed)
        return mystr



#class crustOneModel(Model):
#
#class homogenousModel(Model):
#
#class profileModel(Model):
#
#class IASP91Model(Model):
#
#class AK135Model(Model):
#
#class PREMModel(Model):
#
#class TomographyModel(Model):
