# SimulatorModels.py
#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

# This library is designed to work with the multi-simulator framework by generating 1D, 2D, or 3D models
# The simulators themselves then cast the model output from this into their required format

# Model is an abstraction of a propagation media
# What is it that our elastic or acoustic waves propagate through
# Add: Vs30 for WGAN
# Each simulator class will need methods to ingest an abstract Model
# Differentiate 2D map vs. 2D cross section
# Add inherent dimensionality for Vs30


from abc import ABC, abstractmethod
import numpy as np
from pyfk.tests.taup.test_taup import TestFunctionTaup
import scipy.io


# sets where resources are stored
resourceDir = "data"


class Model(ABC):
    def __init__(self, inherent_dimensionality=None, verbose = False):
        self.inherent_dimensionality = inherent_dimensionality
        self.__initialize_properties__()
        if verbose:
            print("model initialized")
        
    def __initialize_properties__(self):
        self.vp = None
        self.vs = None
        self.vs30 = None
        self.density = None
        self.Qp = None
        self.Qs = None
        self.x = None
        self.y = None
        self.z = None
        self.moho_index = None
        self.layer_thickness = None
        
    def add_test(self):
        print(self.vp)
    
    @property
    def inherent_dimensionality(self):
        return self._inherent_dimensionality
    
    @inherent_dimensionality.setter
    def inherent_dimensionality(self, value):
        self._inherent_dimensionality = value
        
    @property
    def vp(self):
        return self._vp
    
    @vp.setter
    def vp(self, vp_array):
        self._vp = vp_array
        
    @property
    def vs(self):
        return self._vs
    
    @vs.setter
    def vs(self, vs_array):
        self._vs = vs_array
        
    @property
    def density(self):
        return self._density
    
    @property
    def vs30(self):
        return self._vs30
    
    @vs30.setter
    def vs30(self, vs30_array):
        self._vs30 = vs30_array
        
    @density.setter
    def density(self, density_array):
        self._density = density_array
        
    @property
    def Qp(self):
        return self._Qp
    
    @Qp.setter
    def Qp(self, Qp_array):
        self._Qp = Qp_array
        
    @property
    def Qs(self):
        return self._Qs
    
    @Qs.setter
    def Qs(self, Qs_array):
        self._Qs = Qs_array
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x_array):
        self._x = x_array
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y_array):
        self._y = y_array
    
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, z_array):
        self._z = z_array
        
    @property
    def moho_index(self):
        return self._moho_index
    
    @moho_index.setter
    def moho_index(self, moho_index_array):
        self._moho_index = moho_index_array
        
    @property
    def layer_thickness(self):
        return self._layer_thickness
    
    @layer_thickness.setter
    def layer_thickness(self, thick_array):
        self._layer_thickness = thick_array
        
        
    def __str__(self):
        mystr = "Model inherent dimensionality: {}\n".format(self.inherent_dimensionality)
        mystr += "Vp: {}\n".format(self.vp)
        mystr += "Vs: {}\n".format(self.vs)
        mystr += "Vs30: {}\n".format(self.vs30)
        mystr += "Density: {}\n".format(self.density)
        mystr += "Qp: {}\n".format(self.Qp)
        mystr += "Qs: {}\n".format(self.Qs)
        mystr += "X: {}\n".format(self.x)
        mystr += "Y: {}\n".format(self.y)
        mystr += "Z: {}\n".format(self.z)
        mystr += "Moho_index: {}\n".format(self.moho_index)
        mystr += "Thickness: {}\n".format(self.layer_thickness)
        return mystr
    
    def __repr__(self):
        mystr = "Model inherent dimensionality: {}\n".format(self.inherent_dimensionality)
        mystr += "Vp: {}\n".format(self.vp)
        mystr += "Vs: {}\n".format(self.vs)
        mystr += "Vs30: {}\n".format(self.vs30)
        mystr += "Density: {}\n".format(self.density)
        mystr += "Qp: {}\n".format(self.Qp)
        mystr += "Qs: {}\n".format(self.Qs)
        mystr += "X: {}\n".format(self.x)
        mystr += "Y: {}\n".format(self.y)
        mystr += "Z: {}\n".format(self.z)
        mystr += "Moho_index: {}\n".format(self.moho_index)
        mystr += "Thickness: {}\n".format(self.layer_thickness)
        return mystr
        
        
class Model1D(Model):
    def __init__(self, inherent_dimensionality = "1D", verbose=False):
        self.inherent_dimensionality = inherent_dimensionality
        self.__initialize_properties__()
        if verbose:
            print("1D Model initialized")
        
    def taupModel(self, inputFile):
        """
        Reads a .nd file formatted for taup
        """
        try:
            model_data_raw = np.loadtxt(inputFile)
            # generate the model file used for taup
            len_interface = np.shape(model_data_raw)[0]
            model_data = np.zeros((len_interface - 1, 6), dtype=float)
            for index in range(len_interface - 1):
                model_data[index, 0] = model_data_raw[index + 1, 0] - \
                    model_data_raw[index, 0]
                model_data[index, 1] = model_data_raw[index, 2]
                model_data[index, 2] = model_data_raw[index, 1]
                model_data[index, 3] = model_data_raw[index, 3]
                model_data[index, 4] = model_data_raw[index, 5]
                model_data[index, 5] = model_data_raw[index, 4]
            # remove the rows that thickness==0
            model_data = model_data[model_data[:, 0] > 0.05]
            self.layer_thickness = model_data[:,0]
            self.vs = model_data[:,1]
            self.vs30 = model_data[0,1]
            self.vp = model_data[:,2]
            self.density = model_data[:,3]
            self.Qs = model_data[:,4]
            self.Qp = model_data[:,5]
            self.x = None
            self.y = None
            self.z = np.cumsum(self.layer_thickness)
        except:
            print("Error, file {} not found or is not the right format.".format(inputFile))
            
    def PyFK_TaupGenTestModel(self, referenceModel):
        assert referenceModel in ['prem', 'ak135f_no_mud', '1066a'], "Error, PyFK models available are prem, ak135f_no_mud, or 1066a"
        model_data = TestFunctionTaup.gen_test_model(referenceModel)
        self.layer_thickness = model_data[:,0]
        self.vs = model_data[:,1]
        self.vs30 = model_data[0,1]
        self.vp = model_data[:,2]
        self.density = model_data[:,3]
        self.Qs = model_data[:,4]
        self.Qp = model_data[:,5]
        self.x = None
        self.y = None
        self.z = np.cumsum(self.layer_thickness)

    def crustOneModel(self, latitude=None, longitude=None, reference='prem', refmode = "PyFK_TaupGenTest"):
        assert longitude is not None, "Error, longitude must be numerical between -180 and 360."
        assert latitude is not None, "Error, latitude must be numerical between -90 and 90"
        assert longitude >= -180 and longitude <= 360, "Error, longitude must be between -180 and 360"
        assert latitude >= -90 and latitude <= 90, "Error, latitude must be between -90 and 90"
        
        assert refmode in ["PyFK_TaupGenTest", "taupModel", "profileModel", "Existing"], "Error, refmode must be PyFK_TaupGenTest, taupModel, profileModel, or Existing"
        
        mat = scipy.io.loadmat(os.path.join(resourceDir, "Models", "CRUST1.mat"))
        if longitude > 180:
            longitude -= 360
        ilon = int(floor(longitude)+180)-1
        ilat = int(floor(latitude) + 90)-1
        if ilat < 0:
            ilat = 0
        if ilon < 0:
            ilon = 0
        if ilat > 179:
            ilat = 179
        if ilon > 359:
            ilon = 359

        layer_bottoms = mat['C1'][0][0][0][ilat, ilon]
        vp = mat['C1'][0][0][1][ilat, ilon]
        vs = mat['C1'][0][0][2][ilat, ilon]
        if vs[0] == 0.0:
            vs30 = vp[0] / 1.8
        else:
            vs30 = vs[0]
        density = mat['C1'][0][0][3][ilat, ilon]
        
        # Block gets the reference model in the abstract model format
        try:
            if refmode == "PyFK_TaupGenTest":
                self.PyFK_TaupGenTestModel(reference)
            elif refmode == "taupModel":
                self.taupModel(reference)
            elif refmode == "profileModel":
                self.profileModel(reference)
            # Silly, but just a check to make sure we can use the structure in the next blocks
            self.vp *= 1.0
        except:
            print("Error, unable to find or generate the reference model.")
            return
    
        # Now we need to replace all the layers in the reference above the Moho with CRUST1.0 (vp, vs, density, and layer_bottoms)
        # Assuming all is in km depth, positive downward (Crust1.0 is negative downward, so will be flipping those)
        layer_bottoms *= -1.0
        
        
                
        
    
    # homogenous model
    
    # profile model (ie read from file)
    def profileModel(self, inputFile):
        self.PyFK_TaupGenTestModel('prem')
        return
    
    #iasp91, prem, ak135 from different files
    
    # tomography model
    
    
    
# Need to consider 2D map view vs. 2D cross section
class Model2D(Model):
    def __init__(self, inherent_dimensionality = "2D", verbose=False):
        self.inherent_dimensionality = inherent_dimensionality
        self.__initialize_properties__()
        if verbose:
            print("2D Model initialized")
        
    def crustOneModel(self, latitude=None, longitude=None, reference='prem'):
        pass


class Model3D(Model):
    def __init__(self, inherent_dimensionality = "3D", verbose=False):
        self.inherent_dimensionality = inherent_dimensionality
        self.__initialize_properties__()
        if verbose:
            print("3D Model initialized")
    
    def crustOneModel(self, latitude=None, longitude=None, reference='prem'):
        pass
        

    
