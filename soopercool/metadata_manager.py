import soopercool.map_utils as mu
import soopercool.utils as su
import yaml
import numpy as np
import os
import time


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
        self.config = self._yaml_loader(fname_config)

        # Set the high-level parameters as attributes
        for key in self.config:
            setattr(self, key, self.config[key])

        # Set the general attributes (nside, lmax, etc...)
        self._set_general_attributes()

        # Basic sanity checks
        if self.lmax > 3*self.nside-1:
            raise ValueError("lmax should be lower or equal "
                             f"to 3*nside-1 = {3*self.nside-1}")

        # Initialize method to parse map_sets metadata
        map_sets_attributes = list(self.map_sets[
            next(iter(self.map_sets))].keys())
        for map_sets_attribute in map_sets_attributes:
            self._init_getter_from_map_set(map_sets_attribute)

        # A list of the maps used in the analysis
        self.map_sets_list = self._get_map_sets_list()
        self.maps_list = self._get_map_list()

        # Initialize a timer
        self.timer = Timer()

    def _yaml_loader(self, config):
        """
        Custom yaml loader to load the configuration file.
        """
        def path_constructor(loader, node):
            return "/".join(loader.construct_sequence(node))
        yaml.SafeLoader.add_constructor("!path", path_constructor)
        with open(config, "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def _set_directory_attributes(self):
        """
        Set the directory attributes that are listed
        in the paramfiles
        """
        for label, path in self.data_dirs.items():
            if label == "root":
                self.data_dir = self.data_dirs["root"]
            else:
                full_path = f"{self.data_dirs['root']}/{path}"
                setattr(self, label, full_path)
                setattr(self, f"{label}_rel", path)
                os.makedirs(full_path, exist_ok=True)

        for label, path in self.output_dirs.items():
            if label == "root":
                self.output_dir = self.output_dirs["root"]
            else:
                full_path = f"{self.output_dirs['root']}/{path}"
                setattr(self, label, full_path)
                setattr(self, f"{label}_rel", path)
                os.makedirs(full_path, exist_ok=True)

    def _set_general_attributes(self):
        """
        """
        for key, value in self.general_pars.items():
            setattr(self, key, value)

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
                for id_split in range(self.n_bundles_from_map_set(map_set))  # noqa
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
        fname = f"{self.masks['galactic_mask_root']}_{self.masks['gal_mask_mode']}.fits"  # noqa
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
        return os.path.join(self.mask_directory,
                            self.masks["point_source_mask"])

    def _get_analysis_mask_name(self):
        """
        Get the name of the final analysis mask.
        """
        return os.path.join(self.mask_directory, self.masks["analysis_mask"])

    def _get_nhits_map_name(self):
        """
        Get the name of the hits counts map.
        """
        if not self.use_input_nhits:
            # Not using custom nhits map
            return os.path.join(self.mask_directory, self.masks["nhits_map"])
        else:
            # Using custom nhits map
            return self.masks["input_nhits_path"]

    def read_mask(self, mask_type):
        """
        Read the mask given a mask type.

        Parameters
        ----------
        mask_type : str
            Type of mask to load.
            Can be "binary", "galactic", "point_source" or "analysis".
        """
        mask = mu.read_map(getattr(self, f"{mask_type}_mask_name"),
                           pix_type=self.pix_type)
        if self.pix_type == 'hp':
            mask = mu.ud_grade(mask, nside_out=self.nside,
                               pix_type=self.pix_type)
        return mask

    def save_mask(self, mask_type, mask):
        """
        Save the mask given a mask type.

        Parameters
        ----------
        mask_type : str
            Type of mask to load.
            Can be "binary", "galactic", "point_source" or "analysis".
        mask : array-like
            Mask to save.
        """
        return mu.write_map(getattr(self, f"{mask_type}_mask_name"), mask,
                            dtype=np.float32, pix_type=self.pix_type)

    def read_hitmap(self):
        """
        Read the hitmap. For now, we assume that all tags
        share the same hitmap.
        """
        hitmap = mu.read_map(self.nhits_map_name, pix_type=self.pix_type)
        if self.pix_type == 'hp':
            hitmap = mu.ud_grade(hitmap, self.nside, power=-2,
                                 pix_type=self.pix_type)
        return hitmap

    def save_hitmap(self, map):
        """
        Save the hitmap to disk.

        Parameters
        ----------
        map : array-like
            Mask to save.
        """
        mu.write_map(
            os.path.join(self.mask_directory, self.masks["nhits_map"]),
            map, dtype=np.float32, pix_type=self.pix_type)

    def read_nmt_binning(self):
        """
        Read the binning file and return the corresponding NmtBin object.
        """
        import pymaster as nmt
        binning = np.load(self.binning_file)
        return nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)

    def get_n_bandpowers(self):
        """
        Read the binning file and return the number of ell-bins.
        """
        binner = self.read_nmt_binning()
        return binner.get_n_bands()

    def get_effective_ells(self):
        """
        Read the binning file and return the number of ell-bins.
        """
        binner = self.read_nmt_binning()
        return binner.get_effective_ells()

    def read_beam(self, map_set, lmax=None):
        """
        """
        beam_dir = self.beam_dir_from_map_set(map_set)
        beam_file = self.beam_file_from_map_set(map_set)
        l, bl = su.read_beam_from_file(f"{beam_dir}/{beam_file}", lmax=lmax)
        if self.beam_floor is not None:
            bl[bl < self.beam_floor] = self.beam_floor
        return l, bl

    def _init_simulation_params(self):
        """
        Loop over the simulation parameters and set them as attributes.
        """
        for name in ["num_sims", "cosmology", "noise", "anisotropic_noise",
                     "null_e_modes", "mock_nsrcs", "mock_srcs_hole_radius"]:
            setattr(self, name, self.sim_pars[name])

    def _init_filtering_params(self):
        """
        Loop over the filtering parameters and set them as attributes.
        """
        for name in self.filtering:
            setattr(self, name, self.filtering[name])

    def save_fiducial_cl(self, ell, cl_dict, cl_type):
        """
        Save a fiducial power spectra dictionary to disk and return file name.

        Parameters
        ----------
        ell : array-like
            Multipole values.
        cl_dict : dict
            Dictionnary with the power spectra.
        cl_type : str
            Type of power spectra.
            Can be "cosmo", "tf_est", "tf_val" or "noise".
        """
        fname = getattr(self, f"{cl_type}_cls_file")
        np.savez(fname, l=ell, **cl_dict)

    def load_fiducial_cl(self):
        """
        Load fiducial power spectra from healpy-like fits file indicated
        in the parameter file and coadds them.
        Accepted keys are "fiducial_cmb", "fiducial_dust", and "fiducial_synch"
        or a subset thereof. If no file is found, load Planck 2018 camb file
        with r=0 and AL=1.

        Return
        ------
        cl_th: dict
            Unbinned coadded cross-map set power spectra.
        """
        import healpy as hp
        bins = self.read_nmt_binning()
        ps_names = self.get_ps_names_list(type="all", coadd=True)
        cl_theory = {}
        for ms1, ms2 in ps_names:
            cl_theory[ms1, ms2] = None

        if hasattr(self, "fiducial_cmb"):
            if self.fiducial_cmb is not None:
                cmb_cl = hp.read_cl(self.fiducial_cmb)[:, :bins.lmax+1]
                for ps in ps_names:
                    cl_theory[ps] = cmb_cl
        if hasattr(self, "fiducial_dust"):
            if self.fiducial_dust is not None:
                if "{nu1}" not in self.fiducial_dust:
                    raise KeyError("self.fiducial_dust lacks {nu1} marker.")
                if "{nu2}" not in self.fiducial_dust:
                    raise KeyError("self.fiducial_dust lacks {nu2} marker.")
                for ms1, ms2 in ps_names:
                    dust_cl = hp.read_cl(
                        self.fiducial_dust.format(
                            nu1=self.get_freq_tag_from_map_set(ms1),
                            nu2=self.get_freq_tag_from_map_set(ms2)
                        )
                    )[:, :bins.lmax+1]
                if cl_theory[ms1, ms2]:
                    cl_theory[ms1, ms2] += dust_cl
                else:
                    cl_theory = dust_cl
        if hasattr(self, "fiducial_synch"):
            if self.fiducial_synch is not None:
                if "{nu1}" not in self.fiducial_synch:
                    raise KeyError("self.fiducial_synch lacks {nu1} marker.")
                if "{nu2}" not in self.fiducial_synch:
                    raise KeyError("self.fiducial_synch lacks {nu2} marker.")
                for ms1, ms2 in ps_names:
                    synch_cl = hp.read_cl(
                        self.fiducial_synch.format(
                            nu1=self.get_freq_tag_from_map_set(ms1),
                            nu2=self.get_freq_tag_from_map_set(ms2)
                        )
                    )[:, :bins.lmax+1]
                if cl_theory[ms1, ms2]:
                    cl_theory[ms1, ms2] += synch_cl
                else:
                    cl_theory = synch_cl

        if all(x is None for x in cl_theory.values()):
            _, cl_th = su.get_theory_cls()  # Load default theory Cls
            for ps in ps_names:
                cl_theory[ps] = cl_th
        return cl_theory

    def plot_dir_from_output_dir(self, out_dir):
        """
        """
        root = self.output_dir

        if root in out_dir:
            path_to_plots = out_dir.replace(f"{root}/", f"{root}/plots/")
        else:
            path_to_plots = f"{root}/plots/{out_dir}"

        os.makedirs(path_to_plots, exist_ok=True)

        return path_to_plots

    def get_fname_mask(self, map_type='analysis'):
        """
        Get the full filepath to a mask of predefined type.

        Parameters
        ----------
        map_type : str
            Choose between 'analysis', 'binary', 'point_source'.
            Defaults to 'analysis'.
        """
        base_dir = self.masks['mask_directory']
        if map_type == 'analysis':
            fname = os.path.join(base_dir, self.masks['analysis_mask'])
        elif map_type == 'binary':
            fname = os.path.join(base_dir, self.masks['binary_mask'])
        elif map_type == 'point_source':
            fname = os.path.join(base_dir, self.masks['point_source_mask'])
        else:
            raise ValueError("The map_type chosen does not exits. "
                             "Choose between 'analysis', 'binary', "
                             "'point_source'.")
        return fname

    def get_map_filename(self, map_set, id_split, id_sim=None):
        """
        Get the path to file for a given `map_set` and split index.
        Can also get the path to a given simulation if `id_sim` is provided.

        Path to standard map:
            {map_directory}/{map_set_root}_split_{id_split}.fits
        Path to sim map: e.g.
            {sims_directory}/0000/{map_set_root}_split_{id_split}.fits

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

        return os.path.join(path_to_maps,
                            f"{map_set_root}_split_{id_split}.fits")

    def get_filter_function(self, filter_tag):
        from soopercool.utils import m_filter_map, toast_filter_map

        tag_settings = self.filtering["tags_settings"][filter_tag]
        filtering_type = tag_settings["filtering_type"]

        if filtering_type == "m_filterer":
            kwargs = {"m_cut": tag_settings["m_cut"]}
            filter_function = m_filter_map

        elif filtering_type == "toast":
            kwargs = {
                "template": tag_settings["template"],
                "config": tag_settings["config"],
                "schedule": tag_settings["schedule"],
                "instrument": tag_settings["tf_instrument"],
                "band": tag_settings["tf_band"],
                "nside": self.nside,
                "sbatch_dir": self.scripts_dir
            }
            filter_function = toast_filter_map
        else:
            raise NotImplementedError(
                f"Filterer type {tag_settings['filtering_type']} "
                "not implemented"
            )

        def filter_operation(map_file, mask_file, out_dir, extra_kwargs={}):
            return filter_function(
                map_file, mask_file, out_dir, **kwargs, **extra_kwargs)

        return filter_operation

    def print_banner(self, msg):
        """
        print a banner message
        """
        print('')
        print("==============================================================")
        print('')
        print(msg)
        print('')
        print("==============================================================")
        print('')

    def get_map_filename_transfer(self, id_sim, cl_type,
                                  pure_type=None, filter_tag=None):
        """
        """
        path_to_maps = getattr(self, f"{cl_type}_sims_dir")

        beam_label = f"_{filter_tag}" if filter_tag else ""
        pure_label = f"_{pure_type}" if pure_type else ""
        file_name = f"TQU{pure_label}{beam_label}_noiseless_nside{self.nside}_lmax{self.lmax}_{id_sim:04d}.fits"  # noqa

        return f"{path_to_maps}/{file_name}"

    def read_map(self, map_set, id_split, id_sim=None, pol_only=False,
                 convert_K_to_muK=False):
        """
        Read a map given a map set and split index.
        Can also read a given covariance simulation if `id_sim` is provided.

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
        convert_K_to_muK: bool, optional
            Convert maps from K to muK units upon loading.
        """
        field = [1, 2] if pol_only else [0, 1, 2]
        fname = self.get_map_filename(map_set, id_split, id_sim)
        return mu.read_map(fname, field=field, pix_type=self.pix_type,
                           convert_K_to_muK=convert_K_to_muK)

    def read_map_transfer(self, id_sim, signal=None, e_or_b=None,
                          pol_only=False, convert_K_to_muK=False):
        """
        Read a map given a simulation index.
        Can also read a pure-E or pure-B simulation if `e_or_b` is provided.

        Parameters
        ----------
        id_sim : int
            Index of the simulation.
        signal : str, optional
            Determines what signal the validation map corresponds to. For now,
            available choices are 'CMB' and 'dust'.
            If None, assumes simulation mode.
        e_or_b : str
            Accepts either `E`, `B`, or None. Determines what pure-polarization
            type (E or B) the map corresponds to.
            If None, assumes validation mode.
        pol_only : bool, optional
            Return only the polarization maps.
        convert_K_to_muK: bool, optional
            Convert maps from K to muK units upon loading.
        """
        field = [1, 2] if pol_only else [0, 1, 2]
        if (not signal and not e_or_b) or (signal and e_or_b):
            raise ValueError("You have to set to None either `signal` or "
                             "`e_or_b`")
        fname = self.get_map_filename_transfer(id_sim, signal, e_or_b)
        return mu.read_map(fname, field=field, pix_type=self.pix_type,
                           convert_K_to_muK=convert_K_to_muK)

    def get_ps_names_list(self, type="all", coadd=False):
        """
        List the names of cross (and auto) power spectra. Every experiment has
        a number of frequencies and map bundles (splits) with independent
        noise realizations. This function either outputs bundle-coadded
        power spectra (coadd=True), or the cross (and auto) bundle spectra.
        Spectra are considered noise-unbiased if they are either across
        different experiments or across different bundles, or both. Otherwise,
        they have noise bias. Type "cross" selects noise-unbiased spectra,
        type "auto" selects noise-biased spectra, and type "all" selects all.

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
            Can be "all", "auto" or "cross". "auto" returns all unique
            noise-biased spectra, while "cross" returns all unique
            noise-biased spectra. "all" is the union of both.
        coadd: bool, optional
            If True, return the (bundle-)coadded power spectra names.
            If False, return cross (and/or auto) bundle power spectra names.
        """
        map_iterable = self.map_sets_list if coadd else self.maps_list

        ps_name_list = []
        for i, map1 in enumerate(map_iterable):
            for j, map2 in enumerate(map_iterable):

                if coadd:
                    if i > j:
                        continue
                    if (type == "cross") and (i == j):
                        continue
                    if (type == "auto") and (i != j):
                        continue

                else:
                    map_set_1, split_1 = map1.split("__")
                    map_set_2, split_2 = map2.split("__")

                    if (map2, map1) in ps_name_list:
                        continue
                    if (map_set_1 == map_set_2) and (split_1 > split_2):
                        continue

                    exp_tag_1 = self.exp_tag_from_map_set(map_set_1)
                    exp_tag_2 = self.exp_tag_from_map_set(map_set_2)

                    if ((type == "cross") and (exp_tag_1 == exp_tag_2)
                            and (split_1 == split_2)):
                        continue
                    if ((type == "auto") and ((exp_tag_1 != exp_tag_2)
                                              or (split_1 != split_2))):
                        continue

                ps_name_list.append((map1, map2))
        return ps_name_list

    def get_n_split_pairs_from_map_sets(self, map_set_1, map_set_2,
                                        type="cross"):
        """
        Returns the number of unique cross (and auto) bundle spectra that are
        associated to a given pair of map sets ("tagged-coadded" maps).
        Types "cross" or "auto" determine whether to output only the
        noise-unbiased or noise-biased bundle combinations, respectively;
        type "all" returns all of them.

        Example:
            Given two map sets "SAT1_f093" and "SAT1_f145", and 4 bundles
            for SAT1, output the number of splits
            * 4*(4-1)/2 = 6  for type "cross"
            * 4              for type "auto"
            * 4*(4+1)/2 = 10 for type "all"

        Parameters
        ----------
        type : str, optional
            Type of power spectra to return.
            Can be "all", "auto" or "cross". "auto" returns all unique
            noise-biased spectra, while "cross" returns all unique
            noise-biased spectra. "all" is the union of both.
        """
        n_splits_1 = self.n_bundles_from_map_set(map_set_1)
        n_splits_2 = self.n_bundles_from_map_set(map_set_2)
        exp_tag_1 = self.exp_tag_from_map_set(map_set_1)
        exp_tag_2 = self.exp_tag_from_map_set(map_set_2)
        if type == "cross":
            if exp_tag_1 == exp_tag_2:
                n_pairs = n_splits_1 * (n_splits_1 - 1) / 2
            else:
                n_pairs = n_splits_1 * n_splits_2
        elif type == "auto":
            n_pairs = n_splits_1 if exp_tag_1 == exp_tag_2 else 0
        elif type == "all":
            if exp_tag_1 == exp_tag_2:
                n_pairs = n_splits_1 * (n_splits_1 + 1) / 2
            else:
                n_pairs = n_splits_1 * n_splits_2
        else:
            raise ValueError("You selected an invalid type. "
                             "Options are 'cross', 'auto', and 'all'.")
        return n_pairs

    def get_filtering_tags(self):
        """
        """
        return list(set(
            [self.filtering_tag_from_map_set(ms)
             for ms in self.map_sets_list]
        ))

    def get_independent_filtering_pairs(self):
        """
        """
        cross_ps_names = self.get_ps_names_list(coadd=True)
        filtering_pairs = []
        for ms1, ms2 in cross_ps_names:
            fp1 = self.filtering_tag_from_map_set(ms1)
            fp2 = self.filtering_tag_from_map_set(ms2)
            filtering_pairs.append((fp1, fp2))
        return list(set(filtering_pairs))

    def get_inverse_couplings(self, return_bpwf=False):
        """
        This function outputs a dictionary with the filtered and unfiltered
        inverse mode coupling matrices and (optionally,) the bandpower window
        functions.
        """
        couplings_dir = f"{self.output_directory}/couplings"
        filter_labels = {"filtered": "", "unfiltered": "_unfiltered"}
        ps_pairs = self.get_ps_names_list(type="all", coadd=True)
        inv_couplings = {f_type: {} for f_type in filter_labels}
        bandpower_window_functions = {f_type: {} for f_type in filter_labels}

        for ms1, ms2 in ps_pairs:
            ftag1 = self.filtering_tag_from_map_set(ms1)
            ftag2 = self.filtering_tag_from_map_set(ms2)

            for f_type, filter_label in filter_labels.items():
                fname = f"couplings{filter_label}_{ftag1}_{ftag2}"
                if not os.path.isfile(f"{couplings_dir}/{fname}.npz"):
                    raise ValueError(
                        f"Coupling file does not exist: {couplings_dir}/{fname}.npz"  # noqa
                    )

                couplings = np.load(f"{couplings_dir}/{fname}.npz")
                npairs, ndata, _, _ = np.shape(couplings['inv_coupling'])
                c = couplings['inv_coupling'].reshape((npairs*ndata,
                                                       npairs*ndata))
                bpwf = couplings['bp_win']
                bandpower_window_functions[f_type][ftag1, ftag2] = bpwf
                inv_couplings[f_type][ftag1, ftag2] = c

                if ftag1 != ftag2:
                    inv_couplings[f_type][ftag2, ftag1] = c
                    bandpower_window_functions[f_type][ftag2, ftag1] = bpwf

        if not return_bpwf:
            return inv_couplings
        return inv_couplings, bandpower_window_functions

    @classmethod
    def make_dir(cls, dir):
        """
        Make a directory if it does not exist.

        Parameters
        ----------
        dir : str
            Path to the directory.
        """
        os.makedirs(dir, exist_ok=True)


class Timer:
    """
    Basic timer class to time different
    parts of pipeline stages.
    """
    def __init__(self):
        """
        Initialize the timers with an empty dict
        """
        self.timers = {}

    def start(self, timer_label):
        """
        Start the timer with a given label. It allows
        to time multiple nested loops using different labels.

        Parameters
        ----------
        timer_label : str
            Label of the timer.
        """
        if timer_label in self.timers:
            raise ValueError(f"Timer {timer_label} already exists.")
        self.timers[timer_label] = time.time()

    def stop(self, timer_label, text_to_output=None, verbose=True):
        """
        Stop the timer with a given label.
        Allows to output a custom text different
        from the label.

        Parameters
        ----------
        timer_label : str
            Label of the timer.
        text_to_output : str, optional
            Text to output instead of the timer label.
            Defaults to None.
        verbose : bool, optional
            Print the output text.
            Defaults to True.
        """
        if timer_label not in self.timers:
            raise ValueError(f"Timer {timer_label} does not exist.")

        dt = time.time() - self.timers[timer_label]
        self.timers.pop(timer_label)
        if verbose:
            prefix = f"[{text_to_output}]" if text_to_output \
                else f"[{timer_label}]"
            print(f"{prefix} Took {dt:.02f} s to process.")
