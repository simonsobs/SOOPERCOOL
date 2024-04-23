import numpy as np
import os
import soopercool.SO_Noise_Calculator_Public_v3_1_2 as noise_calc
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
import camb


def get_theory_cls(cosmo_params, lmax, lmin=0):
    """
    """
    params = camb.set_params(**cosmo_params)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)
    lth = np.arange(lmin, lmax+1)

    cl_th = {
        "TT": powers["total"][:, 0][lmin:lmax+1],
        "EE": powers["total"][:, 1][lmin:lmax+1],
        "TE": powers["total"][:, 3][lmin:lmax+1],
        "BB": powers["total"][:, 2][lmin:lmax+1]
    }
    for spec in ["EB", "TB"]:
        cl_th[spec] = np.zeros_like(lth)

    return lth, cl_th


def generate_noise_map_white(nside, noise_rms_muKarcmin, ncomp=3):
    """
    """
    size = 12 * nside**2

    pixel_area_deg = hp.nside2pixarea(nside, degrees=True)
    pixel_area_arcmin = 60**2 * pixel_area_deg

    noise_rms_muK_T = noise_rms_muKarcmin / np.sqrt(pixel_area_arcmin)

    out_map = np.zeros((ncomp, size))
    out_map[0, :] = np.random.randn(size) * noise_rms_muK_T

    if ncomp == 3:
        noise_rms_muK_P = np.sqrt(2) * noise_rms_muK_T
        out_map[1, :] = np.random.randn(size) * noise_rms_muK_P
        out_map[2, :] = np.random.randn(size) * noise_rms_muK_P
        return out_map
    return out_map


def get_noise_cls(noise_kwargs, lmax, lmin=0, fsky=0.1,
                  is_beam_deconvolved=False):
    """
    Load polarization noise from SO SAT noise model.
    Assume polarization noise is half of that.
    """
    oof_dict = {"pessimistic": 0, "optimistic": 1}
    oof_mode = noise_kwargs["one_over_f_mode"]
    oof_mode = oof_dict[oof_mode]

    sensitivity_mode = noise_kwargs["sensitivity_mode"]

    noise_model = noise_calc.SOSatV3point1(
        sensitivity_mode=sensitivity_mode,
        N_tubes=[1., 1., 1.],
        one_over_f_mode=oof_mode,
        survey_years=noise_kwargs["survey_years"]
    )
    lth, _, nlth_P = noise_model.get_noise_curves(
        fsky,
        lmax + 1,
        delta_ell=1,
        deconv_beam=is_beam_deconvolved
    )
    lth = np.concatenate(([0, 1], lth))[lmin:]
    nlth_P = np.array(
        [np.concatenate(([0, 0], nl))[lmin:] for nl in nlth_P]
    )

    # Attention: at the moment, the noise model's frequencies must match
    # soopercool's frequency tags.
    freq_tags = [int(f) for f in noise_model.get_bands()]
    nl_all_frequencies = {}
    for i_f, freq_tag in enumerate(freq_tags):
        nl_th_dict = {pq: nlth_P[i_f]
                      for pq in ["EE", "EB", "BE", "BB"]}
        nl_th_dict["TT"] = 0.5*nlth_P[i_f]
        nl_th_dict["TE"] = 0.*nlth_P[i_f]
        nl_th_dict["TB"] = 0.*nlth_P[i_f]
        nl_all_frequencies[freq_tag] = nl_th_dict

    return lth, nl_all_frequencies


def generate_noise_map(nl_T, nl_P, hitmap, n_splits, is_anisotropic=True):
    """
    """
    # healpix ordering ["TT", "EE", "BB", "TE"]
    noise_mat = np.array([nl_T, nl_P, nl_P, np.zeros_like(nl_P)])
    # Normalize the noise
    noise_mat *= n_splits

    noise_map = hp.synfast(noise_mat, hp.get_nside(hitmap), pol=True, new=True)

    if is_anisotropic:
        # Weight with hitmap
        noise_map[:, hitmap != 0] /= np.sqrt(hitmap[hitmap != 0] / np.max(hitmap)) # noqa

    return noise_map


def random_src_mask(mask, nsrcs, mask_radius_arcmin):
    """
    pspy.so_map
    """
    ps_mask = mask.copy()
    src_ids = np.random.choice(np.where(mask == 1)[0], nsrcs)
    for src_id in src_ids:
        vec = hp.pix2vec(hp.get_nside(mask), src_id)
        disc = hp.query_disc(hp.get_nside(mask), vec,
                             np.deg2rad(mask_radius_arcmin / 60))
        ps_mask[disc] = 0
    return ps_mask


def get_beam_windows(meta, plot=False):
    """
    Compute and save dictionary with beam window functions for each map set.
    """
    oof_dict = {"pessimistic": 0, "optimistic": 1}

    noise_model = noise_calc.SOSatV3point1(
        survey_years=meta.noise["survey_years"],
        sensitivity_mode=meta.noise["sensitivity_mode"],
        one_over_f_mode=oof_dict[meta.noise["one_over_f_mode"]]
    )

    lth = np.arange(3*meta.nside)
    beam_arcmin = {int(freq_band): beam_arcmin
                   for freq_band, beam_arcmin in zip(noise_model.get_bands(),
                                                     noise_model.get_beams())}
    beams_dict = {}
    for map_set in meta.map_sets_list:
        if "SAT" not in meta.exp_tag_from_map_set(map_set):
            continue
        freq_tag = meta.freq_tag_from_map_set(map_set)
        beams_dict[map_set] = beam_gaussian(lth, beam_arcmin[freq_tag])
        file_root = meta.file_root_from_map_set(map_set)

        if not os.path.exists(file_root):
            np.savetxt(f"{meta.beam_directory}/beam_{file_root}.dat",
                       np.transpose([lth, beams_dict[map_set]]))
        if plot:
            plt.plot(lth, beams_dict[map_set], label=map_set)
    if plot:
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"{meta.beam_directory}/beams.png")


def beam_gaussian(ll, fwhm_amin):
    """
    Returns the SHT of a Gaussian beam.
    Args:
        l (float or array): multipoles.
        fwhm_amin (float): full-widht half-max in arcmins.
    Returns:
        float or array: beam sampled at `l`.
    """
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * ll * (ll + 1) * sigma_rad**2)


def beam_hpix(ll, nside):
    """
    Returns the SHT of the beam associated with a HEALPix
    pixel size.
    Args:
        l (float or array): multipoles.
        nside (int): HEALPix resolution parameter.
    Returns:
        float or array: beam sampled at `l`.
    """
    fwhm_hp_amin = 60 * 41.7 / nside
    return beam_gaussian(ll, fwhm_hp_amin)


def create_binning(nside, delta_ell, end_first_bin=None):
    """
    """
    if end_first_bin is not None:
        bin_low = np.arange(end_first_bin, 3*nside, delta_ell)
        bin_high = bin_low + delta_ell - 1
        bin_low = np.concatenate(([0], bin_low))
        bin_high = np.concatenate(([end_first_bin-1], bin_high))
    else:
        bin_low = np.arange(0, 3*nside, delta_ell)
        bin_high = bin_low + delta_ell - 1
    bin_high[-1] = 3*nside - 1
    bin_center = (bin_low + bin_high) / 2

    return bin_low, bin_high, bin_center


def power_law_cl(ell, amp, delta_ell, power_law_index):
    """
    """
    pl_ps = {}
    for spec in ["TT", "TE", "TB", "EE", "EB", "BB"]:
        if isinstance(amp, dict):
            A = amp[spec]
        else:
            A = amp
        # A is power spectrum amplitude at pivot ell == 1 - delta_ell
        pl_ps[spec] = A / (ell + delta_ell) ** power_law_index

    return pl_ps


def m_filter_map(map_file, mask_file, out_dir, m_cut):
    """
    Applies the m-cut mock filter to a given map with a given sky mask.

    Parameters
    ----------
    map : array-like
        Healpix TQU map to be filtered.
    map_file : str
        File path of the unfiltered map.
    mask : array-like
        Healpix map storing the sky mask.
    m_cut : int
        Maximum nonzero m-degree of the multipole expansion. All higher
        degrees are set to zero.
    """

    map = hp.read_map(map_file, field=(0, 1, 2))
    mask = hp.read_map(mask_file)
    mask[mask != 0] = 1.

    map_masked = map * mask
    nside = hp.get_nside(map)
    lmax = 3 * nside - 1

    alms = hp.map2alm(map_masked, lmax=lmax)

    n_modes_to_filter = (m_cut + 1) * (lmax + 1) - ((m_cut + 1) * m_cut) // 2
    alms[:, :n_modes_to_filter] = 0.

    filtered_map = hp.alm2map(alms, nside=nside, lmax=lmax)

    fname = os.path.basename(map_file)
    fname_out = fname.replace(".fits", "_filtered.fits")

    hp.write_map(f"{out_dir}/{fname_out}",
                 filtered_map, overwrite=True,
                 dtype=np.float32)


def m_filter_map_old(map, map_file, mask, m_cut):
    """
    Applies the m-cut mock filter to a given map with a given sky mask.

    Parameters
    ----------
    map : array-like
        Healpix TQU map to be filtered.
    map_file : str
        File path of the unfiltered map.
    mask : array-like
        Healpix map storing the sky mask.
    m_cut : int
        Maximum nonzero m-degree of the multipole expansion. All higher
        degrees are set to zero.
    """

    map_masked = map * mask
    nside = hp.get_nside(map)
    lmax = 3 * nside - 1

    alms = hp.map2alm(map_masked, lmax=lmax)

    n_modes_to_filter = (m_cut + 1) * (lmax + 1) - ((m_cut + 1) * m_cut) // 2
    alms[:, :n_modes_to_filter] = 0.

    filtered_map = hp.alm2map(alms, nside=nside, lmax=lmax)

    hp.write_map(map_file.replace('.fits', '_filtered.fits'),
                 filtered_map, overwrite=True,
                 dtype=np.float32)


def toast_filter_map(map, map_file, mask,
                     template, config, schedule,
                     nside, instrument, band,
                     sbatch_job_name, sbatch_dir,
                     nhits_map_only=False, sim_noise=False):
    """
    Create sbatch scripts for each simulation, based on given template file.

    Parameters
    ----------
    map : array-like (unused)
        This is an unused argument included for compatibility with other
        filters. TOAST won't read the map itself.
    map_file : str
        File path of the unfiltered map.
    mask : array-like (unused)
        This is an unused argument included for compatibility with other
        filters. TOAST won't read the mask itself.
    template : str
        Path to sbatch template file in Jinja2.
    config : str
        Path to TOAST toml config file.
    schedule : str
        Path to TOAST schedule file.
    nside : int
        Healpix Nside parameter of the filtered map.
    instrument : str
        Name of the instrument simulated by TOAST.
    band : str
        Name of the frequency band simulated by TOAST.
    sbatch_job_name : str
        Sbatch job name
    sbatch_dir : str
        Sbatch output directory.
    nhits_map_only : bool
        If True, only get a hits map from TOAST schedule file.
    sim_noise : bool
        If True, simulate noise with TOAST.
    """
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path

    del map, mask  # delete unused arguments

    # Path(...).resovle() will return absolute path.
    map_file = Path(map_file).resolve()
    if nhits_map_only:
        map_dir = map_file.parent
        map_dir.mkdir(parents=True, exist_ok=True)
    template_file = Path(template).resolve()
    template_dir = template_file.parent
    template_name = template_file.name
    config_file = Path(config).resolve()
    schedule_file = Path(schedule).resolve()
    sbatch_dir = Path(sbatch_dir).resolve()
    sbatch_outdir = sbatch_dir/sbatch_job_name
    sbatch_outdir.mkdir(parents=True, exist_ok=True)
    sbatch_file = sbatch_dir/(sbatch_job_name + '.sh')
    sbatch_log = sbatch_dir/(sbatch_job_name + '.log')

    jinja2_env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True)
    jinja2_temp = jinja2_env.get_template(template_name)

    with open(sbatch_file, mode='w') as f:
        f.write(jinja2_temp.render(
            sbatch_job_name=sbatch_job_name,
            sbatch_log=sbatch_log,
            outdir=str(sbatch_outdir),
            nside=nside,
            band=band,
            telescope=instrument,
            config=str(config_file),
            schedule=str(schedule_file),
            map_file=str(map_file),
            nhits_map_only=nhits_map_only,
            sim_noise=sim_noise,
            ))
    os.chmod(sbatch_file, 0o755)
    return sbatch_file


def get_split_pairs_from_coadd_ps_name(map_set1, map_set2,
                                       all_splits_ps_names,
                                       cross_splits_ps_names,
                                       auto_splits_ps_names):
    """
    """
    split_pairs_list = {
        "auto": [],
        "cross": []
    }
    for split_ms1, split_ms2 in all_splits_ps_names:
        if (not (split_ms1.startswith(map_set1) and
                 split_ms2.startswith(map_set2))):
            continue

        if (split_ms1, split_ms2) in cross_splits_ps_names:
            split_pairs_list["cross"].append((split_ms1, split_ms2))
        elif (split_ms1, split_ms2) in auto_splits_ps_names:
            split_pairs_list["auto"].append((split_ms1, split_ms2))

    return split_pairs_list


def plot_map(map, fname, vrange_T=300, vrange_P=10, title=None, TQU=True):
    fields = "TQU" if TQU else "QU"
    for i, m in enumerate(fields):
        vrange = vrange_T if m == "T" else vrange_P
        plt.figure(figsize=(16, 9))
        hp.mollview(map[i], title=f"{title}_{m}", unit=r'$\mu$K$_{\rm CMB}$',
                    cmap=cm.coolwarm, min=-vrange, max=vrange)
        hp.graticule()
        plt.savefig(f"{fname}_{m}.png", bbox_inches="tight")


def beam_alms(alms, bl):
    """
    """
    if bl is not None:
        for i, alm in enumerate(alms):
            alms[i] = hp.almxfl(alm, bl)

    return alms


def generate_map_from_alms(alms, nside, pureE=False, pureB=False,
                           pureT=False, bl=None):
    """
    """
    alms = beam_alms(alms, bl)
    Tlm, Elm, Blm = alms
    if pureE:
        alms = [Tlm*0., Elm, Blm*0.]
    elif pureB:
        alms = [Tlm*0., Elm*0., Blm]
    elif pureT:
        alms = [Tlm, Elm*0., Blm*0.]

    return hp.alm2map(alms, nside, lmax=3*nside - 1)


def bin_validation_power_spectra(cls_dict, nmt_binning,
                                 bandpower_window_function):
    """
    Bin multipoles of transfer function validation power spectra into
    binned bandpowers.
    """
    nl = nmt_binning.lmax + 1
    cls_binned_dict = {}

    for spin_comb in ["spin0xspin0", "spin0xspin2", "spin2xspin2"]:
        bpw_mat = bandpower_window_function[f"bp_win_{spin_comb}"]

        for val_type in ["tf_val", "cosmo"]:
            if spin_comb == "spin0xspin0":
                cls_vec = np.array([cls_dict[val_type]["TT"][:nl]])
                cls_vec = cls_vec.reshape(1, nl)
            elif spin_comb == "spin0xspin2":
                cls_vec = np.array([cls_dict[val_type]["TE"][:nl],
                                    cls_dict[val_type]["TB"][:nl]])
            elif spin_comb == "spin2xspin2":
                cls_vec = np.array([cls_dict[val_type]["EE"][:nl],
                                    cls_dict[val_type]["EB"][:nl],
                                    cls_dict[val_type]["EB"][:nl],
                                    cls_dict[val_type]["BB"][:nl]])

            cls_vec_binned = np.einsum("ijkl,kl", bpw_mat, cls_vec)

            if spin_comb == "spin0xspin0":
                cls_binned_dict[val_type, "TT"] = cls_vec_binned[0]
            elif spin_comb == "spin0xspin2":
                cls_binned_dict[val_type, "TE"] = cls_vec_binned[0]
                cls_binned_dict[val_type, "TB"] = cls_vec_binned[1]
            elif spin_comb == "spin2xspin2":
                cls_binned_dict[val_type, "EE"] = cls_vec_binned[0]
                cls_binned_dict[val_type, "EB"] = cls_vec_binned[1]
                cls_binned_dict[val_type, "BE"] = cls_vec_binned[2]
                cls_binned_dict[val_type, "BB"] = cls_vec_binned[3]

    return cls_binned_dict


def plot_transfer_function(lb, tf_dict, lmin, lmax, field_pairs, file_name):
    """
    Plot the transfer function given an input dictionary.
    """
    plt.figure(figsize=(25, 25))
    grid = plt.GridSpec(9, 9, hspace=0.3, wspace=0.3)

    for id1, f1 in enumerate(field_pairs):
        for id2, f2 in enumerate(field_pairs):
            ax = plt.subplot(grid[id1, id2])

            ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)

            ax.errorbar(
                lb, tf_dict[f"{f1}_to_{f2}"], tf_dict[f"{f1}_to_{f2}_std"],
                marker=".", markerfacecolor="white",
                color="navy")

            if id1 == 8:
                ax.set_xlabel(r"$\ell$", fontsize=14)
            else:
                ax.set_xticks([])

            if f1 == f2:
                ax.axhline(1., color="k", ls="--")
            else:
                ax.axhline(0, color="k", ls="--")
                ax.ticklabel_format(axis="y", style="scientific",
                                    scilimits=(0, 0), useMathText=True)

            ax.set_xlim(lmin, lmax)

    plt.savefig(file_name, bbox_inches="tight")


def plot_transfer_validation(meta, map_set_1, map_set_2,
                             cls_theory, cls_theory_binned,
                             cls_mean_dict, cls_std_dict):
    """
    Plot the transfer function validation power spectra and save to disk.
    """
    nmt_binning = meta.read_nmt_binning()
    lb = nmt_binning.get_effective_ells()

    for val_type in ["tf_val", "cosmo"]:
        plt.figure(figsize=(16, 16))
        grid = plt.GridSpec(9, 3, hspace=0.3, wspace=0.3)

        for id1, id2 in [(i, j) for i in range(3) for j in range(3)]:
            f1, f2 = "TEB"[id1], "TEB"[id2]
            spec = f2 + f1 if id1 > id2 else f1 + f2

            main = plt.subplot(grid[3*id1:3*(id1+1)-1, id2])
            sub = plt.subplot(grid[3*(id1+1)-1, id2])

            # Plot theory
            ell = cls_theory[val_type]["l"]
            rescaling = 1 if val_type == "tf_val" \
                else ell * (ell + 1) / (2*np.pi)
            main.plot(ell, rescaling*cls_theory[val_type][spec], color="k")

            offset = 0.5
            rescaling = 1 if val_type == "tf_val" else lb*(lb + 1) / (2*np.pi)

            # Plot filtered & unfiltered (decoupled)
            if not meta.validate_beam:
                main.errorbar(
                    lb - offset, rescaling*cls_mean_dict[val_type,
                                                         "unfiltered",
                                                         spec],
                    rescaling*cls_std_dict[val_type, "unfiltered", spec],
                    color="navy", marker=".", markerfacecolor="white",
                    label=r"Unfiltered decoupled $C_\ell$", ls="None"
                )
            main.errorbar(
                lb + offset, rescaling*cls_mean_dict[val_type,
                                                     "filtered",
                                                     spec],
                rescaling*cls_std_dict[val_type, "filtered", spec],
                color="darkorange", marker=".", markerfacecolor="white",
                label=r"Filtered decoupled $C_\ell$", ls="None"
            )

            if f1 == f2:
                main.set_yscale("log")

            # Plot residuals
            sub.axhspan(-2, 2, color="gray", alpha=0.2)
            sub.axhspan(-1, 1, color="gray", alpha=0.7)
            sub.axhline(0, color="k")

            if not meta.validate_beam:
                residual_unfiltered = (
                    (cls_mean_dict[val_type, "unfiltered", spec]
                     - cls_theory_binned[val_type, spec])
                    / cls_std_dict[val_type, "unfiltered", spec]
                )
                sub.plot(
                    lb - offset,
                    residual_unfiltered * np.sqrt(meta.tf_est_num_sims),
                    color="navy", marker=".", markerfacecolor="white",
                    ls="None"
                )
            residual_filtered = (
                (cls_mean_dict[val_type, "filtered", spec]
                 - cls_theory_binned[val_type, spec])
                / cls_std_dict[val_type, "filtered", spec]
            )
            sub.plot(lb + offset,
                     residual_filtered * np.sqrt(meta.tf_est_num_sims),
                     color="darkorange", marker=".",
                     markerfacecolor="white", ls="None")

            # Multipole range
            main.set_xlim(2, meta.lmax)
            sub.set_xlim(*main.get_xlim())

            # Suplot y range
            sub.set_ylim((-5., 5.))

            # Cosmetix
            main.set_title(f1+f2, fontsize=14)
            if spec == "TT":
                main.legend(fontsize=13)
            main.set_xticklabels([])
            if id1 != 2:
                sub.set_xticklabels([])
            else:
                sub.set_xlabel(r"$\ell$", fontsize=13)

            if id2 == 0:
                if isinstance(rescaling, float):
                    main.set_ylabel(r"$C_\ell$", fontsize=13)
                else:
                    main.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$",
                                    fontsize=13)
                sub.set_ylabel(r"$\Delta C_\ell / (\sigma/\sqrt{N_\mathrm{sims}})$",  # noqa
                               fontsize=13)

        plot_dir = meta.plot_dir_from_output_dir(meta.coupling_directory)
        plot_suffix = (f"__{map_set_1}_{map_set_2}" if meta.validate_beam
                       else "")
        plt.savefig(f"{plot_dir}/decoupled_{val_type}{plot_suffix}.pdf",
                    bbox_inches="tight")


def get_binary_mask_from_nhits(nhits_map, nside, zero_threshold=1e-3):
    """
    Make binary mask by smoothing, normalizing and thresholding nhits map.
    """
    nhits_smoothed = hp.smoothing(
        hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64),
        fwhm=np.pi/180)
    nhits_smoothed[nhits_smoothed < 0] = 0
    nhits_smoothed /= np.amax(nhits_smoothed)
    binary_mask = np.zeros_like(nhits_smoothed)
    binary_mask[nhits_smoothed > zero_threshold] = 1

    return binary_mask


def get_apodized_mask_from_nhits(nhits_map, nside,
                                 galactic_mask=None,
                                 point_source_mask=None,
                                 zero_threshold=1e-3,
                                 apod_radius=10.,
                                 apod_radius_point_source=4.,
                                 apod_type="C1"):
    """
    Produce an appropriately apodized mask from an nhits map as used in
    the BB pipeline paper (https://arxiv.org/abs/2302.04276).

    Procedure:
    * Make binary mask by smoothing, normalizing and thresholding nhits map
    * (optional) multiply binary mask by galactic mask
    * Apodize (binary * galactic)
    * (optional) multiply (binary * galactic) with point source mask
    * (optional) apodize (binary * galactic * point source)
    * Multiply everything by (smoothed) nhits map
    """
    import pymaster as nmt

    # Smooth and normalize hits map
    nhits_map = hp.smoothing(
        hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64),
        fwhm=np.pi/180)
    nhits_map /= np.amax(nhits_map)

    # Get binary mask
    binary_mask = get_binary_mask_from_nhits(nhits_map, nside, zero_threshold)

    # Multiply by Galactic mask
    if galactic_mask is not None:
        binary_mask *= hp.ud_grade(galactic_mask, nside)

    # Apodize the binary mask
    binary_mask = nmt.mask_apodization(binary_mask, apod_radius,
                                       apotype=apod_type)

    # Multiply with point source mask
    if point_source_mask is not None:
        binary_mask *= hp.ud_grade(point_source_mask, nside)
        binary_mask = nmt.mask_apodization(binary_mask,
                                           apod_radius_point_source,
                                           apotype=apod_type)

    return nhits_map * binary_mask


def get_spin_derivatives(map):
    """
    First and second spin derivatives of a given spin-0 map.
    """
    nside = hp.npix2nside(np.shape(map)[-1])
    ell = np.arange(3*nside)
    alpha1i = np.sqrt(ell*(ell + 1.))
    alpha2i = np.sqrt((ell - 1.)*ell*(ell + 1.)*(ell + 2.))
    first = hp.alm2map(hp.almxfl(hp.map2alm(map), alpha1i), nside=nside)
    second = hp.alm2map(hp.almxfl(hp.map2alm(map), alpha2i), nside=nside)
    cmap = cm.YlOrRd
    cmap.set_under("w")

    return first, second
