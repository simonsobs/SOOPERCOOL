import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap


def get_coupled_pseudo_cls(fields1, fields2, nmt_binning,
                           return_unbinned=False):
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
    if return_unbinned:
        pcls_unbinned = {}
    for spin1 in spins:
        for spin2 in spins:

            f1 = fields1[spin1]
            f2 = fields2[spin2]

            coupled_cell = nmt.compute_coupled_cell(f1, f2)
            coupled_cell = coupled_cell[:, :nmt_binning.lmax+1]

            pcls[f"{spin1}x{spin2}"] = nmt_binning.bin_cell(coupled_cell)
            if return_unbinned:
                pcls_unbinned[f"{spin1}x{spin2}"] = coupled_cell
    if return_unbinned:
        return pcls, pcls_unbinned

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


def get_weighted_pcls(pcls, mask, pix_type="car"):
    """
    Returns the weighted coupled pseudo-C_ells PCL = PCL / w where

    w = sum_{pixels p} Omega[p] / (4*pi) * mask[p]**2

    and Omega[p] is the pixel size in steradians.

    Args
    ----
        coupled_pseudo_cells : dict with keys f"spin{s1}xspin{s2}",
            items array-like. Coupled pseudo-C_ell estimators.
        mask: np.array or enmap.ndarray
            Mask defining the mode coupling.
    Returns
    -------
        dict with keys f"spin{s1}xspin{s2}", items array-like.
        Coupled pseudo-C_ell estimators.
    """
    pcls_dict = field_pairs_from_spins(pcls)

    if pix_type == "hp":
        weights = np.mean(mask ** 2)
    elif pix_type == "car":
        shape, wcs = mask.geometry
        pixsizemap = enmap.pixsizemap(shape, wcs)  # steradians
        weights = np.sum(mask ** 2 * pixsizemap) / (4*np.pi)

    for k in pcls_dict:
        pcls_dict[k] = pcls_dict[k] / weights

    return pcls_dict


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


def get_pcls_mat_transfer(fields, nmt_binning, fields2=None,
                          return_unbinned=False):
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

    if return_unbinned:
        pcls_mat_unbinned = np.zeros((9, 9, nmt_binning.lmax+1))
        tmp_pcls_unbinned = {}

    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    tmp_pcls = {}
    for (pure_type1, pure_type2) in field_pairs:
        pure_type1 = f"pure{pure_type1}"
        pure_type2 = f"pure{pure_type2}"
        pcls = get_coupled_pseudo_cls(
            fields[pure_type1],
            fields2[pure_type2],
            nmt_binning,
            return_unbinned=return_unbinned
        )
        if return_unbinned:
            pcls, pcls_unbinned = pcls
            tmp_pcls_unbinned[pure_type1, pure_type2] = {
                "TT": pcls_unbinned["spin0xspin0"][0],
                "TE": pcls_unbinned["spin0xspin2"][0],
                "TB": pcls_unbinned["spin0xspin2"][1],
                "EE": pcls_unbinned["spin2xspin2"][0],
                "EB": pcls_unbinned["spin2xspin2"][1],
                "BE": pcls_unbinned["spin2xspin2"][2],
                "BB": pcls_unbinned["spin2xspin2"][3]
            }

        tmp_pcls[pure_type1, pure_type2] = {
            "TT": pcls["spin0xspin0"][0],
            "TE": pcls["spin0xspin2"][0],
            "TB": pcls["spin0xspin2"][1],
            "EE": pcls["spin2xspin2"][0],
            "EB": pcls["spin2xspin2"][1],
            "BE": pcls["spin2xspin2"][2],
            "BB": pcls["spin2xspin2"][3]
        }

    for idx, (pure_type1, pure_type2) in enumerate(field_pairs):
        pure_type1 = f"pure{pure_type1}"
        pure_type2 = f"pure{pure_type2}"
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

        if return_unbinned:
            pcls_mat_unbinned[idx] = np.array([
                tmp_pcls_unbinned[pure_type1, pure_type2]["TT"],
                tmp_pcls_unbinned[pure_type1, pure_type2]["TE"],
                tmp_pcls_unbinned[pure_type1, pure_type2]["TB"],
                tmp_pcls_unbinned[pure_type2, pure_type1]["TE"],
                tmp_pcls_unbinned[pure_type2, pure_type1]["TB"],
                tmp_pcls_unbinned[pure_type1, pure_type2]["EE"],
                tmp_pcls_unbinned[pure_type1, pure_type2]["EB"],
                tmp_pcls_unbinned[pure_type1, pure_type2]["BE"],
                tmp_pcls_unbinned[pure_type1, pure_type2]["BB"]
            ])

    if return_unbinned:
        return pcls_mat, pcls_mat_unbinned
    return pcls_mat


def read_nmt_binning(binning_file, lmax, compute_Dl, force_cl=False):
    """
    Read the binning file and return the corresponding NmtBin object.
    """
    binning = np.load(binning_file)
    bin_low, bin_high = binning["bin_low"], binning["bin_high"]
    bin_lmax = bin_high[-1]

    if bin_lmax < lmax:
        raise ValueError(
            f"lmax in binning {bin_lmax} is lower than {lmax}."
            " Update config file to change lmax or binning scheme."
        )
    else:
        # Truncate binning
        select = bin_low < lmax
        bin_low = bin_low[select]
        bin_high = bin_high[select]

        bin_high = np.concatenate((
            bin_high[:-1],
            np.array([lmax])
        ))

    return nmt.NmtBin.from_edges(
        bin_low,
        bin_high + 1,
        is_Dell=compute_Dl if not force_cl else False
    )


def bin_theory_cls(cls, bpwf):
    """
    Convolve theory power spectra with bandpower window function.

    Args
    ----
        cls: dict, items strings with field pairs "TT", "TE", etc, 
            values array-like of shape (nl,), where `nl` is the maximum
             multipole
        bpwf: ndarray
            Binned power window functions of shape (size, n_bins, size, nl).
            `size` is the number of field combinations (e.g., 9 for TEBxTEB),
            and `n_bins` is the number of bandpower bins.
    Returns
    -------
        dict, items strings with field pairs "TT", "TE", etc, 
        values array-like of shape (nl,), where `nl` is the maximum
        multipole
    """
    fields_all = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    nl_th = cls["TT"].shape[0]

    size, n_bins, _, nl = bpwf.shape
    assert size == 9, "Unexpected number of fields in coupling matrix"
    assert nl <= nl_th, f"Theory spectrum must contain ell up to {nl}."

    cls_dict = {}
    for fp in fields_all:
        if fp in cls:
            cls_dict[fp] = cls[fp][:nl]
        # E.g., if "TE" is in cls but "ET" is not, set "ET" equal to "TE".
        elif fp[1]+fp[0] in cls:
            cls_dict[fp] = cls[fp[1]+fp[0]][:nl]
        # E.g., if neither "EB" nor "BE" are in cls, set them both to zero.
        else:
            cls_dict[fp] = np.zeros(nl)
    cls_vec = np.array([cls_dict[fp] for fp in fields_all])

    clb = np.dot(bpwf.reshape(size*n_bins, size*nl), cls_vec.reshape(size*nl))
    clb = clb.reshape(size, n_bins)

    return {fp: clb[ifp] for ifp, fp in enumerate(fields_all)}


def plot_pcls_mat_transfer(pcls_mat_unfilt, pcls_mat_filt, lb, file_name,
                           lmax=None):
    """
    Plot the pure-type coupled and binned power spectra which are the inputs of
    the SOOPERCOOL transfer function if tf_ordering is set to "TM".

    Args
    ----
        pcls_mat_unfilt: np.ndarray
            PCL matrix of unfiltered simulations.
            Array of shape (size, size, nl), where
            `size` is the number of field combinations (e.g., 9 for TEBxTEB),
            and `n_bins` is the number of bandpower bins. The first axis loops
            over the output field pairs, the second axis loops over the
            pure-type pairs.
        pcls_mat_filt: np.ndarray
            PCL matrix of filtered simulations.
        lb: array-like, shape (n_bins,)
            Effective bandpowers of the pcl matrices.
        file_name: string
            File name for the output plot.
        lmax: int
            Maximum multiplot to plot.
    """
    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    plt.figure(figsize=(25, 25))
    grid = plt.GridSpec(9, 9, hspace=0.3, wspace=0.3)
    msk = np.ones_like(lb).astype(bool)
    if lmax is not None:
        msk = lb <= lmax

    for id1, f1 in enumerate(field_pairs):
        for id2, f2 in enumerate(field_pairs):
            ax = plt.subplot(grid[id1, id2])
            ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)
            ax.plot(lb[msk], pcls_mat_unfilt[id1, id2][msk],
                    c="navy", label="unfiltered")
            ax.plot(lb[msk], -pcls_mat_unfilt[id1, id2][msk],
                    c="navy", ls="--")
            ax.plot(lb[msk], pcls_mat_filt[id1, id2][msk],
                    c="darkorange", label="filtered")
            ax.plot(lb[msk], -pcls_mat_filt[id1, id2][msk],
                    c="darkorange", ls="--")
            ax.set_yscale("log")
            if id1 == 8:
                ax.set_xlabel(r"$\ell$", fontsize=14)
            else:
                ax.set_xticks([])
            if (id1, id2) == (0, 0):
                ax.legend(fontsize=14)

    plt.savefig(file_name, bbox_inches="tight")


def plot_spectrum(lb, cb, cb_err, title, ylabel, xlim,
                  cb_data=None, cb_data_err=None, add_theory=False,
                  lth=None, clth=None, cbth=None, save_file=None):
    """
    Helper function to plot power spectra.
    TODO: This is only used in sacc_plotter.py. Generalise, if needed, and
    extend its use to other scripts.
    """
    plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(4, 1, wspace=0, hspace=0)
    if add_theory:
        main = plt.subplot(grid[:-1])
        sub = plt.subplot(grid[-1])
        sub.set_xlabel(r"$\ell$")
        sub.set_ylabel(r"$\Delta C_\ell / (\sigma / \sqrt{N_\mathrm{sims}})$")
    else:
        main = plt.subplot(grid[:])
        main.set_xlabel(r"$\ell$")

    main.set_ylabel(r"$\ell (\ell + 1) C_\ell^{%s} / 2\pi$" % ylabel)
    main.set_title(title)

    fac = lb * (lb + 1) / (2 * np.pi)
    offset = 0 if cb_data is None else 1

    if add_theory:
        fac_th = lth * (lth + 1) / (2 * np.pi)
        main.plot(lth, fac_th * clth, c="darkgray", ls="-.", lw=2.6,
                  label="theory")

    main.errorbar(
        lb-offset, fac * cb, yerr=fac * cb_err, marker="o", ls="None",
        markerfacecolor="white", markeredgecolor="navy", label="sims",
        elinewidth=1.75, ecolor="navy", markeredgewidth=1.75
    )

    if cb_data is not None:
        main.errorbar(
            lb+offset, fac * cb_data, yerr=fac * cb_data_err, marker="o",
            ls="None", markerfacecolor="white", markeredgecolor="darkorange",
            elinewidth=1.75, ecolor="darkorange", markeredgewidth=1.75,
            label="data"
        )

    main.legend(fontsize=13)
    main.set_xlim(*xlim)

    if add_theory:
        sub.axhspan(-3, 3, color="gray", alpha=0.1)
        sub.axhspan(-2, 2, color="gray", alpha=0.4)
        sub.axhspan(-1, 1, color="gray", alpha=0.8)

        sub.plot(
            lb-offset, (cb - cbth) / cb_err, marker="o", ls="None",
            markerfacecolor="white", markeredgecolor="navy",
            markeredgewidth=1.75
        )
        if cb_data is not None:
            sub.plot(
                lb+offset, (cb_data - cbth) / cb_data_err,
                marker="o", ls="None",
                markerfacecolor="white", markeredgecolor="darkorange",
                markeredgewidth=1.75
            )

        sub.set_xlim(*xlim)
        sub.set_ylim(-4.5, 4.5)

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()
    plt.close()
