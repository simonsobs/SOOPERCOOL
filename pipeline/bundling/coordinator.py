#!/bin/sh python3

import numpy as np
import healpy as hp
import h5py


_RA_Dec = np.array([[-60.1,-44.1,-28.1,-12.1,4.0,20.0,36.0,-4.0,12.1,28.1,44.1,60.1,76.1,-52.3,-36.2,-20.2,-4.2,11.8,27.9,-12.1,3.9,19.9,36.0,52.0,68.0,84.0,-60.4,-44.4,-28.4,-12.4,3.7,19.7,35.7,-4.2,11.8,27.8,43.8,59.8,75.8,-52.5,-36.5,-20.5,-4.5,11.5,27.6,-12.4,3.6,19.7,35.7,51.7,67.7,83.7,-60.7,-44.7,-28.7,-12.6,3.4,19.4,35.4,-4.5,11.5,27.5,43.5,59.6,75.6,-52.8,-36.8,-20.8,-4.8,11.3,27.3,-12.7,3.3,19.4,35.4,51.4,67.4,83.4,99.4,-61.0,-45.0,-28.9,-12.9,3.1,19.1,35.2,-4.8,11.2,27.2,43.3,59.3,75.3,91.3,107.3,123.2,-53.1,-37.1,-21.1,-5.1,11.0,27.0,-13.0,3.1,19.1,35.1,51.1,67.1,83.1,99.1,115.1,-61.3,-45.3,-29.2,-13.2,2.8,18.8,34.9,-5.1,10.9,26.9,43.0,59.0,75.0,107.0,123.0,147.5,-53.4,-37.4,-21.4,-5.3,10.7,26.7,-13.3,2.8,18.8,34.8,50.8,66.8,82.8,114.8,128.8,-93.5,-77.6,-61.6,-45.5,-29.5,-13.5,2.5,18.6,34.6,-5.4,10.6,26.7,42.7,58.7,74.7,122.7,-85.7,-69.7,-53.7,-37.7,-21.7,-5.6,10.4,26.4,-13.5,2.5,18.5,34.5,50.6,66.6,82.6,-93.8,-77.8,-61.8,-45.8,-29.8,-13.8,2.2,18.3,34.3,-5.7,10.4,26.4,42.4,58.4,74.4,-86.0,-70.0,-54.0,-38.0,-21.9,-5.9,10.1,26.1,-13.8,2.2,18.2,34.2,50.3,66.3,82.3,-62.1,-46.1,-30.1,-14.1,2.0,18.0,34.0,-6.0,10.1,26.1,42.1,58.1,74.1,-54.3,-38.2,-22.2,-6.2,9.8,25.9,-14.1,1.9,17.9,34.0,50.0,66.0,82.0,-46.4,-30.4,-14.4,1.7,17.7,33.7,-6.2,9.8,25.8,41.8,57.8,73.8,-22.5,-6.5,9.5,25.6,-14.4,1.6,17.7,33.7,49.7,65.7,81.7,-14.6,1.4,17.4,33.4,-6.5,9.5,25.5,41.5,57.6,73.6,9.3,25.3,41.3,1.3,17.4,33.4,49.4,65.4,81.4,17.1,33.2,49.2,9.2,25.2,41.3,57.3,73.3,-55.1,-39.1,-23.1,-63.0,-47.0,25.0,41.0,1.1,17.1,33.1,49.1,65.1,81.1,-47.3,-31.2,-15.2,0.8,16.8,32.9,-7.1,8.9,24.9,41.0,57.0,73.0,-55.4,-39.4,-23.4,-7.3,8.7,24.7,40.7,0.8,16.8,32.8,48.8,64.9,80.9,-47.5,-31.5,-15.5,0.5,16.6,32.6,-7.4,8.6,24.7,40.7,56.7,72.7,-55.7,-39.7,-23.7,-7.6,8.4,24.4,40.4,0.5,16.5,32.5,48.6,64.6,80.6,-47.8,-31.8,-15.8,0.2,16.3,32.3,-7.7,8.4,24.4,40.4,56.4,72.4,-56.0,-40.0,-23.9,-7.9,8.1,24.1,40.2,0.2,16.2,32.2,48.3,64.3,80.3,-48.1,-32.1,-16.1,-0.0,16.0,32.0,-8.0,8.1,24.1,40.1,56.1,72.1,-56.3,-40.2,-24.2,-8.2,7.8,23.9,39.9,-0.1,15.9,32.0,48.0,64.0,80.0,-48.4,-32.4,-16.4,-0.3,15.7,31.7,-8.2,7.8,23.8,39.8,55.8,71.9,-56.5,-40.5,-24.5,-8.5,7.5,23.6,39.6,-0.4,15.7,31.7,47.7,63.7,79.7,-48.7,-32.7,-16.6,-0.6,15.4,31.4,-8.5,7.5,23.5,39.5,55.6,71.6,87.6,103.6,-56.8,-40.8,-24.8,-8.8,7.3,23.3,39.3,-0.7,15.4,31.4,47.4,63.4,79.4,95.4,111.4,-49.0,-32.9,-16.9,-0.9,15.1,31.2,-8.8,7.2,23.2,39.3,55.3,71.3,87.3,103.3,119.3,-57.1,-41.1,-25.1,-9.1,7.0,23.0,39.0,-0.9,15.1,31.1,47.1,63.1,79.1,95.1,111.1,-97.2,-49.2,-33.2,-17.2,-1.2,14.8,30.9,-9.1,6.9,22.9,39.0,55.0,71.0,87.0,103.0,119.0,-89.4,-57.4,-41.4,-25.4,-9.3,6.7,22.7,38.7,-1.2,14.8,30.8,46.8,62.9,78.9,-129.5,-97.5,-81.5,-49.5,-33.5,-17.5,-1.5,14.6,30.6,-9.4,6.6,22.7,38.7,54.7,70.7,86.7,-89.7,-73.7,-57.7,-41.7,-25.6,-9.6,6.4,22.4,38.4,-1.5,14.5,30.5,46.6,62.6,78.6,-81.8,-65.8,-49.8,-33.8,-17.8,-1.8,14.3,30.3,-9.7,6.4,22.4,38.4,54.4,70.4,86.4,-74.0,-58.0,-42.0,-25.9,-9.9,6.1,22.1,38.2,-1.8,14.2,30.3,46.3,62.3,78.3,-66.1,-50.1,-34.1,-18.1,-2.0,14.0,30.0,-10.0,6.1,22.1,38.1,54.1,70.1,86.1,-42.2,-26.2,-10.2,5.8,21.9,37.9,-2.1,13.9,30.0,46.0,62.0,78.0,-34.4,-18.4,-2.3,13.7,29.7,-10.2,5.8,21.8,37.8,53.8,69.9,85.9,-10.5,5.5,21.6,37.6,-2.4,13.7,29.7,45.7,61.7,77.7,-2.6,13.4,29.4,-10.5,5.5,21.5,37.5,53.6,69.6,85.6,-58.8,5.3,21.3,37.3,-2.7,13.4,29.4,45.4,61.4,77.4,-51.0,-34.9,-58.9,13.1,29.2,45.2,5.2,21.2,37.3,53.3,69.3,85.3,-59.1,-43.1,-27.1,-11.1,5.0,21.0,37.0,-2.9,13.1,29.1,45.1,61.1,77.1,-51.2,-35.2,-19.2,-3.2,12.8,28.9,-11.1,4.9,20.9,37.0,53.0,69.0,85.0,-59.4,-43.4,-27.4,-11.3,4.7,20.7,36.7,-3.2,12.8,28.8,44.8,60.9,76.9,-51.5,-35.5,-19.5,-3.5,12.6,28.6,-11.4,4.6,20.7,36.7,52.7,68.7,84.7,-59.7,-43.7,-27.6,-11.6,4.4,20.4,36.4,-3.5,12.5,28.5,44.6,60.6,76.6,-51.8,-35.8,-19.8,-3.8,12.3,28.3,-11.7,4.4,20.4,36.4,52.4,68.4,84.4,-60.0,-43.9,-27.9,-11.9,4.1,20.1,36.2,-3.8,12.2,28.3,44.3,60.3,76.3,-52.1,-36.1,-20.1,-4.0,12.0,28.0,-12.0,4.1,20.1,36.1,52.1,68.1,84.1,-60.2,-44.2,-28.2,-12.2,3.8,19.9,35.9,-4.1,11.9,28.0,44.0,60.0,76.0,-52.4,-36.4,-20.3,-4.3,11.7,27.7,-12.2,3.8,19.8,35.8,51.8,67.9,83.9,-60.5,-44.5,-28.5,-12.5,3.5,19.6,35.6,-4.4,11.7,27.7,43.7,59.7,75.7,91.7,-52.7,-36.7,-20.6,-4.6,11.4,27.4,-12.5,3.5,19.5,35.5,51.6,67.6,83.6,99.6,-60.8,-44.8,-28.8,-12.8,3.3,19.3,35.3,-4.7,11.4,27.4,43.4,59.4,75.4,91.4,107.4,123.4,-53.0,-36.9,-20.9,-4.9,11.1,27.2,-12.8,3.2,19.2,35.3,51.3,67.3,83.3,99.3,115.3,-61.1,-45.1,-29.1,-13.1,3.0,19.0,35.0,-4.9,11.1,27.1,43.1,59.1,75.2,107.1,123.1,-85.2,-53.2,-37.2,-21.2,-5.2,10.8,26.9,-13.1,2.9,19.0,35.0,51.0,67.0,83.0,115.0,-93.4,-77.4,-61.4,-45.4,-29.4,-13.3,2.7,18.7,34.7,-5.2,10.8,26.8,42.8,58.9,74.9,122.8,-117.5,-85.5,-69.5,-53.5,-37.5,-21.5,-5.5,10.6,26.6,-13.4,2.6,18.7,34.7,50.7,66.7,82.7,-93.7,-77.7,-61.7,-45.7,-29.6,-13.6,2.4,18.4,34.5,-5.5,10.5,26.5,42.6,58.6,74.6,-85.8,-69.8,-53.8,-37.8,-21.8,-5.8,10.3,26.3,-13.7,2.4,18.4,34.4,50.4,66.4,82.4,-62.0,-45.9,-29.9,-13.9,2.1,18.1,34.2,-5.8,10.2,26.3,42.3,58.3,74.3,-54.1,-38.1,-22.1,-6.0,10.0,26.0,-14.0,2.1,18.1,34.1,50.1,66.1,82.2,-30.2,-14.2,1.8,17.9,33.9,-6.1,9.9,26.0,42.0,58.0,74.0,-22.3,-6.3,9.7,25.7,-14.2,1.8,17.8,33.8,49.8,65.9,81.9,1.5,17.6,33.6,-6.4,9.7,25.7,41.7,57.7,73.7,9.4,25.4,41.5,1.5,17.5,33.5,49.6,65.6,81.6,-46.8,-54.7,17.3,33.3,49.3,9.4,25.4,41.4,],
                    [-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.2,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.2,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.2,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-38.9,-38.9,-39.1,-39.1,-39.1,-39.2,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-38.9,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-38.9,-38.9,-38.9,-39.1,-39.1,-39.1,-39.2,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-38.9,-38.9,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-38.9,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-38.9,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-38.9,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-38.9,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-38.9,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-38.9,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-38.9,-38.9,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-38.9,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-38.9,-38.9,-39.0,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-38.9,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-38.9,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.2,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,-39.0,-39.0,-39.1,-39.0,-39.1,-39.1,-39.1,-39.1,-39.1,-39.1,]])


def get_list_of_RA(deg=True):
    if deg:
        return _RA_Dec[0]
    else:
        return np.deg2rad(_RA_Dec[0])


def get_list_of_Dec(deg=True):
    if deg:
        return _RA_Dec[1]
    else:
        return np.deg2rad(_RA_Dec[1])


def read_hdf5_map(fname, to_nest=False):
    f = h5py.File(fname, "r")
    dset = f["map"]
    header = dict(dset.attrs)
    nside_file = header["NSIDE"]

    if header["ORDERING"] == "NESTED":
        file_nested = True
    elif header["ORDERING"] == "RING":
        file_nested = False

    nnz, npix = dset.shape
    nside = hp.npix2nside(npix)

    if file_nested and not to_nest:
        mapdata = hp.reorder(dset[:], n2r=True)
    elif not file_nested and to_nest:
        mapdata = hp.reorder(dset[:], r2n=True)
    else:
        mapdata = dset

    return mapdata



def write_hdf5_map(fname, nside, dict_maps, list_of_obsid, nest_or_ring='RING'):
    with h5py.File(fname, 'w') as f:
        f.attrs['NSIDE'] = nside
        f.attrs['ORDERING'] = nest_or_ring
        f.attrs['OBS_ID'] = list_of_obsid
        for k,v in dict_maps.items():
                f.create_dataset(k, data=v)


def gen_masks_of_given_atomic_map_list_for_bundles(nmaps, nbundles):
    # add one more atomic map to bundles from the beginning until the remainers have gone
    n_per_bundle = nmaps // nbundles
    nremainder = nmaps % nbundles

    masks = []
    for idx in range(nbundles):
        if idx < nremainder:
            _n_per_bundle = n_per_bundle + 1
        else:
            _n_per_bundle = n_per_bundle

        i_begin = idx * _n_per_bundle
        i_end = (idx+1) * _n_per_bundle

        _m = np.zeros(nmaps, dtype=np.bool_)
        _m[i_begin:i_end] = True
        
        masks.append(_m)

    return masks
