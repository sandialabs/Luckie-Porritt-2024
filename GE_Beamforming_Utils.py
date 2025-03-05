#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

from pykml import parser
import numpy as np
import geoUtils as gu

def get_coords_from_kml(filename): 
    """
    Given a filename of a kml file, reads contents and returns a list of coordinates
    """

    with open(filename) as f:
        contents = parser.parse(f)

    coord_string = contents.getroot().Document.Placemark.LineString.coordinates

    tmp_string = str(coord_string)
    coord_string = tmp_string.split(" ")

    for idx, mystr in enumerate(coord_string):
        coord_string[idx] = mystr.strip()
 
    # Last one appears to be junk
    coord_string = coord_string[:-1]

    output = np.zeros((len(coord_string), 3))

    for idx, mystr in enumerate(coord_string):
        output[idx, 0] = float(mystr.split(",")[0])
        output[idx, 1] = float(mystr.split(",")[1])
        output[idx, 2] = float(mystr.split(",")[2])

    return output

def convert_coords_to_utm(coords, demean=True):
    """
    Uses the geoUtils function deg2utm to convert from floating point degrees to utm
    Use output from "get_coords_from_kml" as input coords
    Returns demeaned x and y vectors if demean=True
    """
    x, y, z = gu.deg2utm(coords[:,1], coords[:,0])
    if demean:
        x = x - x.mean()
        y = y - y.mean()
    return x, y, z

def infill_das_sampling(x, y, channel_separation = 1.0):
    """
    Populates x_out and y_out arrays with intermediary points at channel_separation offset between spatial coordinate vectors x and y
    Note that it assumes x and y are in meters.
    """

    assert len(x) == len(y), "Error, x and y must be the same length"
    nseg = len(x) - 1
    
    # For each segment, get the angle between the ith and (i+1)th point
    x_out = []
    y_out = []
    for iseg in range(nseg):
        dy = y[iseg+1] - y[iseg]
        dx = x[iseg+1] - x[iseg]
        angle = np.arctan2(dy, dx)
        distance = np.sqrt(dx**2 + dy**2)
        n_new_pts = int(np.floor(distance/channel_separation))
        #xtmp = np.zeros((n_new_pts,))
        #ytmp = np.zeros((n_new_pts,))
        for idx in range(n_new_pts):
            x_out.append(x[iseg] + idx*channel_separation * np.cos(angle))
            y_out.append(y[iseg] + idx*channel_separation * np.sin(angle)) 
            #xtmp[idx] = x[iseg] + channel_separation * np.cos(angle)
            #ytmp[idx] = y[iseg] + channel_separation * np.sin(angle)
        #x_out.append(xtmp)
        #y_out.append(ytmp)

    x_out = np.asarray(x_out)
    y_out = np.asarray(y_out)
    return x_out, y_out

