from itertools import product
import pymaster as nmt
import numpy as np


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
    for spin_comb, coupled_pcl in coupled_pseudo_cells.items():
        n_bins = coupled_pcl.shape[-1]
        decoupled_pcl = coupling_inv[spin_comb] @ coupled_pcl.flatten()
        if spin_comb == "spin0xspin0":
            size = 1
        elif spin_comb in ["spin0xspin2", "spin2xspin0"]:
            size = 2
        elif spin_comb == "spin2xspin2":
            size = 4
        decoupled_pcl = decoupled_pcl.reshape((size, n_bins))

        decoupled_pcls[spin_comb] = decoupled_pcl

    decoupled_pcls = field_pairs_from_spins(decoupled_pcls)

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
    pcls_mat_00 = np.zeros((1, 1, n_bins))
    pcls_mat_02 = np.zeros((2, 2, n_bins))
    pcls_mat_22 = np.zeros((4, 4, n_bins))

    index = 0
    cases = ["pureE", "pureB"]
    for index, (pure_type1, pure_type2) in enumerate(product(cases, cases)):
        pcls = get_coupled_pseudo_cls(fields[pure_type1],
                                      fields2[pure_type2],
                                      nmt_binning)
        pcls_mat_22[index] = pcls["spin2xspin2"]
        pcls_mat_02[cases.index(pure_type2)] = pcls["spin0xspin2"]

    pcls_mat_00[0] = pcls["spin0xspin0"]

    return {"spin0xspin0": pcls_mat_00,
            "spin0xspin2": pcls_mat_02,
            "spin2xspin2": pcls_mat_22}
