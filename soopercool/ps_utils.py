import soopercool.map_utils as mu
from itertools import product
import pymaster as nmt
import numpy as np


def get_validation_power_spectra(meta, id_sim, mask, nmt_binning,
                                 inv_couplings):
    """
    This function computes transfer validation power spectra given an
    input simulation ID, mask and binning scheme, and stores them to disk.
    """
    map_set_pairs = (meta.get_ps_names_list(type="all", coadd=True)
                     if meta.validate_beam else [(None, None)])
    filter_flags = (["filtered"] if meta.validate_beam
                    else ["filtered", "unfiltered"])

    for cl_type in ["tf_val", "cosmo"]:
        for filter_flag in filter_flags:
            for map_sets in map_set_pairs:
                map_files = [
                    meta.get_map_filename_transfer2(
                        id_sim, cl_type=cl_type, map_set=ms
                    ) for ms in map_sets
                ]

                if filter_flag == "filtered":
                    map_files = [mf.replace(".fits", "_filtered.fits")
                                 for mf in map_files]

                maps = [mu.read_map(m, field=[0, 1, 2], convert_K_to_muK=True)
                        for m in map_files]

                field = [{
                    "spin0": nmt.NmtField(mask, map[:1]),
                    "spin2": nmt.NmtField(mask, map[1:],
                                          purify_b=meta.tf_est_pure_B)
                } for map in maps]

                pcls = get_coupled_pseudo_cls(field[0], field[1], nmt_binning)

                if meta.validate_beam:
                    decoupled_pcls = decouple_pseudo_cls(
                        pcls, inv_couplings[map_sets[0], map_sets[1]]
                    )
                else:
                    decoupled_pcls = decouple_pseudo_cls(
                        pcls, inv_couplings[filter_flag]
                    )
                cl_prefix = f"pcls_{cl_type}_{id_sim:04d}"
                cl_suffix = (f"_{map_sets[0]}_{map_sets[1]}"
                             if meta.validate_beam else f"_{filter_flag}")
                cl_name = cl_prefix + cl_suffix

                np.savez(f"{meta.cell_transfer_directory}/{cl_name}.npz",
                         **decoupled_pcls)


def get_binned_cls(bp_win_dict, cls_dict_unbinned):
    """
    """
    nl = np.shape(list(bp_win_dict.values())[0])[-1]
    cls_dict_binned = {}

    for spin_comb in ["spin0xspin0", "spin0xspin2", "spin2xspin2"]:
        bpw_mat = bp_win_dict[f"bp_win_{spin_comb}"]
        if spin_comb == "spin0xspin0":
            cls_vec = np.array([cls_dict_unbinned["TT"][:nl]]).reshape(1, nl)
        elif spin_comb == "spin0xspin2":
            cls_vec = np.array([cls_dict_unbinned["TE"][:nl],
                                cls_dict_unbinned["TB"][:nl]])
        elif spin_comb == "spin2xspin2":
            cls_vec = np.array([cls_dict_unbinned["EE"][:nl],
                                cls_dict_unbinned["EB"][:nl],
                                cls_dict_unbinned["EB"][:nl],
                                cls_dict_unbinned["BB"][:nl]])

        cls_dict_binned[spin_comb] = np.einsum("ijkl,kl", bpw_mat, cls_vec)

    return field_pairs_from_spins(cls_dict_binned)


def get_coupled_pseudo_cls(fields1, fields2, nmt_binning):
    """
    Compute the binned coupled pseudo-C_ell estimates from two
    (spin-0 or spin-2) NaMaster fields and a multipole binning scheme.
    Parameters
    ----------
    fields1, fields2 : NmtField
        Spin-0 or spin-2 fields to correlate.
    nmt_binning : NmtBin
        Multipole binning scheme.
    """
    spins = list(fields1.keys())

    pcls = {}
    for spin1 in spins:
        for spin2 in spins:

            f1 = fields1[spin1]
            f2 = fields2[spin2]

            coupled_cell = nmt.compute_coupled_cell(f1, f2)
            coupled_cell = coupled_cell[:, :nmt_binning.lmax+1]

            pcls[f"{spin1}x{spin2}"] = nmt_binning.bin_cell(coupled_cell)
    return pcls


def decouple_pseudo_cls(coupled_pseudo_cells, coupling_inv):
    """
    Decouples the coupled pseudo-C_ell estimators computed between two fields
    of spin 0 or 2. Returns decoupled binned power spectra labeled by field
    pairs (e.g. 'TT', 'TE', 'EE', 'EB', 'BB' etc.).
    Parameters
    ----------
    coupled_pseudo_cells : dict with keys f"spin{s1}xspin{s2}",
        items array-like. Coupled pseudo-C_ell estimators.
    coupling_inv : array-like
        Inverse binned bandpower coupling matrix.
    """
    decoupled_pcls = {}
    stacked_pcls = np.concatenate(
        np.vstack([
            coupled_pseudo_cells["spin0xspin0"],
            coupled_pseudo_cells["spin0xspin2"],
            coupled_pseudo_cells["spin2xspin0"],
            coupled_pseudo_cells["spin2xspin2"]
        ])
    )
    decoupled_pcls_vec = coupling_inv @ stacked_pcls

    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    nbins = coupled_pseudo_cells["spin0xspin0"].shape[-1]
    for i, fp in enumerate(field_pairs):
        decoupled_pcls[fp] = decoupled_pcls_vec[i*nbins:(i+1)*nbins]
    return decoupled_pcls


def field_pairs_from_spins(cls_in_dict):
    """
    Reorders power spectrum dictionary with a given input spin
    pair into pairs of output (pseudo-)scalar fields on the sky
    (T, E, or B).

    Parameters
    ----------
    cls_in_dict: dictionary
    """
    cls_out_dict = {}

    field_spin_mapping = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin0": ["ET", "BT"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }

    for spin_pair in cls_in_dict:
        for index, field_pair in enumerate(field_spin_mapping[spin_pair]):

            cls_out_dict[field_pair] = cls_in_dict[spin_pair][index]

    return cls_out_dict


def get_pcls_mat_transfer(fields, nmt_binning, fields2=None):
    """
    Compute coupled binned pseudo-C_ell estimates from
    pure-E and pure-B transfer function estimation simulations,
    and cast them into matrix shape.

    Parameters
    ----------
    fields: dictionary of NmtField objects (keys "pureE", "pureB")
    nmt_binning: NmtBin object
    fields2: dict, optional
        If not None, compute the pseudo-C_ell estimators
        from the cross-correlation of the fields in `fields`
        and `fields2`.
    """
    if fields2 is None:
        fields2 = fields

    n_bins = nmt_binning.get_n_bands()
    pcls_mat = np.zeros((9, 9, n_bins))

    index = 0
    cases = ["pureT", "pureE", "pureB"]
    tmp_pcls = {}
    for index, (pure_type1, pure_type2) in enumerate(product(cases, cases)):
        pcls = get_coupled_pseudo_cls(fields[pure_type1],
                                      fields2[pure_type2],
                                      nmt_binning)
        tmp_pcls[pure_type1, pure_type2] = {
            "TT": pcls["spin0xspin0"][0],
            "TE": pcls["spin0xspin2"][0],
            "TB": pcls["spin0xspin2"][1],
            "EE": pcls["spin2xspin2"][0],
            "EB": pcls["spin2xspin2"][1],
            "BE": pcls["spin2xspin2"][2],
            "BB": pcls["spin2xspin2"][3]
        }

    for idx, (pure_type1, pure_type2) in enumerate(product(cases, cases)):
        pcls_mat[idx] = np.array([
            tmp_pcls[pure_type1, pure_type2]["TT"],
            tmp_pcls[pure_type1, pure_type2]["TE"],
            tmp_pcls[pure_type1, pure_type2]["TB"],
            tmp_pcls[pure_type2, pure_type1]["TE"],
            tmp_pcls[pure_type2, pure_type1]["TB"],
            tmp_pcls[pure_type1, pure_type2]["EE"],
            tmp_pcls[pure_type1, pure_type2]["EB"],
            tmp_pcls[pure_type1, pure_type2]["BE"],
            tmp_pcls[pure_type1, pure_type2]["BB"]
        ])

    return pcls_mat
