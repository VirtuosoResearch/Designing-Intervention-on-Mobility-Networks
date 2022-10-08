import os
import io
import zipfile
import json
import scipy
from scipy import sparse
from numpy.random.mtrand import beta
import pandas as pd
import numpy as np
from shapely.geometry import Polygon  
import shapely.wkt
from pyproj import Geod

# import all directory constants
from helper_constants import *

def load_static_networks(msa_name):
    network = sparse.load_npz(MONTHLY_NETWORK_DIR(msa_name) + "weights.npz")
    beta_poi = np.load(MONTHLY_NETWORK_DIR(msa_name) + "beta_poi.npy")
    cbg_population = np.load(MONTHLY_NETWORK_DIR(msa_name) + "cbg_population.npy")
    return network, beta_poi, cbg_population

"""
Unzips nested zipfiles into a folder structure reflective 
of the original nested zipfile structure.
"""
def nested_unzip(path):
    z = zipfile.ZipFile(path)

    # iterate through sub-zipfiles
    for f in z.namelist():

        # create a directory for each file's path root (i.e. folder name)
        dirname = os.path.splitext(f)[0]
        os.mkdir(dirname)

        # read inner zip file into bytes buffer
        content = io.BytesIO(z.read(f))
        zip_file = zipfile.ZipFile(content)

        # iterate through zipped files and extract them
        for i in zip_file.namelist():
            zip_file.extract(i, dirname)
