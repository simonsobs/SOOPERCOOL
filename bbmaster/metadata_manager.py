import yaml
import numpy as np
import os
from scipy.interpolate import interp1d
import pymaster as nmt
import healpy as hp
import sacc
from itertools import combinations_with_replacement as cwr
from itertools import combinations
import camb

class BBmeta(object):
    """
    Metadata manager for the BBmaster pipeline.
    The purpose of this class is to provide
    a single interface to all the parameters and products
    that will be used from different stages of the pipeline.
    """


    def __init__(self, fname_config):
        """
        Initialize the pipeline manager from a yaml file.
        
        Parameters
        ----------
        fname_config : str
            Path to the yaml file with the configuration.
        """

        # Load the configuration file
        with open(fname_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Set the high-level parameters as attributes
        for key in self.config:
            setattr(self, key, self.config[key])
        
        # Basic sanity checks
        if self.lmax > 3*self.nside-1:
            raise ValueError(f"lmax should be lower or equal to 3*nside-1 = {3*self.nside-1}")
        
        # Set all the `_directory` attributes
        self._set_directory_attributes()

        # Initialize method to parse map_sets metadata
        map_sets_attributes = list(self.map_sets[next(iter(self.map_sets))].keys())
        for map_sets_attribute in map_sets_attributes:
            self._init_getter_from_map_set(map_sets_attribute)

        # A list of the maps used in the analysis
        self.map_sets_list = self._get_map_sets_list()
        self.maps_list = self._get_map_list()

        # Initialize masks file_names
        for mask_type in ["binary", "galactic", "point_source", "analysis"]:
            setattr(
                self,
                f"{mask_type}_mask_name", 
                getattr(self, f"_get_{mask_type}_mask_name")()
            )


    def _set_directory_attributes(self):
        """
        Set the directory attributes that are listed
        in the paramfiles
        """
        for value in self.config.values():
            # Skip if value is not a dict (e.g. lmax, nside, ...)
            if not isinstance(value, dict): continue
            # Loop over 2nd layer dict to find directory definitions
            for subkey, subvalue in value.items():
                if subkey.endswith("_directory"):
                    setattr(self, subkey, subvalue)


    def _get_map_sets_list(self):
        """
        List the different map set.
        Constructor for the map_sets_list attribute.
        """
        return list(self.map_sets.keys())
    
    
    def _get_map_list(self):
        """
        List the different maps (including splits).
        Constructor for the map_list attribute.
        """
        out_list = [
            f"{map_set}__{id_split}" for map_set in self.map_sets_list
                for id_split in range(self.n_splits_from_map_set(map_set))
        ]
        return out_list


    def _init_getter_from_map_set(self, map_set_attribute):
        """
        Initialize a getter for a map_set attribute.

        Parameters
        ----------
        map_set_attribute : str
            Should a key of the map_set dictionnary.
        """
        setattr(
            self,
            f"{map_set_attribute}_from_map_set",
            lambda map_set: self.map_sets[map_set][map_set_attribute]
        )


    def _get_galactic_mask_name(self):
        """
        Get the name of the galactic mask.
        """
        fname = f"{self.masks['galactic_mask_root']}_{self.masks['gal_mask_mode']}.fits"
        return os.path.join(self.mask_directory, fname)


    def _get_binary_mask_name(self):
        """
        Get the name of the binary or survey mask.
        """
        return os.path.join(self.mask_directory, self.masks["binary_mask"])


    def _get_point_source_mask_name(self):
        """
        Get the name of the point source mask.
        """
        return os.path.join(self.mask_directory, self.masks["point_source_mask"])


    def _get_analysis_mask_name(self):
        """
        Get the name of the final analysis mask.
        """
        return os.path.join(self.mask_directory, self.masks["analysis_mask"])


    def read_mask(self, mask_type):
        """
        Read the mask given a mask type.

        Parameters
        ----------
        mask_type : str
            Type of mask to load.
            Can be "binary", "galactic", "point_source" or "analysis".
        """
        return hp.read_map(getattr(self, f"{mask_type}_mask_name"))


    def save_mask(self, mask_type, mask, overwrite=False):
        """
        Save the mask given a mask type.

        Parameters
        ----------
        mask_type : str
            Type of mask to load.
            Can be "binary", "galactic", "point_source" or "analysis".
        mask : array-like
            Mask to save.
        overwrite : bool, optional
            Overwrite the mask if it already exists.
        """
        return hp.write_map(getattr(self, f"{mask_type}_mask_name"), mask, overwrite=overwrite)

    
    def get_map_filename(self, map_set, id_split, id_sim=None):
        """
        Get the path to file for a given `map_set` and split index.
        Can also get the path to a given simulation if `id_sim` is provided.

        Path to standard map : {map_directory}/{map_set_root}_split_{id_split}.fits
        Path to sim map : e.g. {sims_directory}/0000/{map_set_root}_split_{id_split}.fits

        Parameters
        ----------
        map_set : str
            Name of the map set.
        id_split : int
            Index of the split.
        id_sim : int, optional
            Index of the simulation.
            If None, return the path to the data map.
        """
        map_set_root = self.file_root_from_map_set(map_set)
        if id_sim is not None:
            path_to_maps = os.path.join(
                self.sims_directory,
                f"{id_sim:04d}"
            )
            os.makedirs(path_to_maps, exist_ok=True)
        else:
            path_to_maps = self.map_directory
        
        return os.path.join(path_to_maps, f"{map_set_root}_split_{id_split}.fits")
    
    def read_map(self, map_set, id_split, id_sim=None, pol_only=False):
        """
        Read a map given a map set and split index.
        Can also read a given simulation if `id_sim` is provided.

        Parameters
        ----------
        map_set : str
            Name of the map set.
        id_split : int
            Index of the split.
        id_sim : int, optional
            Index of the simulation.
            If None, return the data map.
        pol_only : bool, optional
            Return only the polarization maps.
        """
        field = [1, 2] if pol_only else [0, 1, 2]
        fname = self.get_map_filename(map_set, id_split, id_sim=id_sim)
        return hp.read_map(fname, field=field)
    
    def get_ps_names_list(self, type="all", coadd=False):
        """
        List all the possible cross split power spectra
        Example: 
            From two map_sets `ms1` and `ms2` with
            two splits each, and `type="all"`, 
            this function will return :
            [('ms1__0', 'ms1__0'), ('ms1__0', 'ms1__1'),
             ('ms1__0', 'ms2__0'), ('ms1__0', 'ms2__1'),
             ('ms1__1', 'ms1__1'), ('ms1__1', 'ms2__0'),
             ('ms1__1', 'ms2__1'), ('ms2__0', 'ms2__0'),
             ('ms2__0', 'ms2__1'), ('ms2__1', 'ms2__1')]

        Parameters
        ----------
        type : str, optional
            Type of power spectra to return.
            Can be "all", "auto" or "cross".
        coadd: bool, optional
            If True, return the cross-split power spectra names.
            Else, return the (split-)coadded power spectra names.
        """
        map_iterable = self.map_sets_list if coadd else self.maps_list
        if type == "all":
            return list(cwr(map_iterable, 2))
        elif type == "auto":
            return [(map_name, map_name) for map_name in map_iterable]
        elif type == "cross":
            return list(combinations(map_iterable, 2))