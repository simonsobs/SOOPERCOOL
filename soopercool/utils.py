import yaml
import numpy as np
import os
from scipy.interpolate import interp1d
import soopercool.SO_Noise_Calculator_Public_v3_1_2 as noise_calc
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
import sacc
import camb


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
    import pymaster as nmt
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


def get_pcls(man, fnames, names, fname_out, mask, binning, winv=None):
    """
    man -> pipeline manager
    fnames -> files with input maps
    names -> map names
    fname_out -> output file name
    mask -> mask
    binning -> binning scheme to use
    winv -> inverse binned MCM (optional)
    """
    import pymaster as nmt

    if winv is not None:
        nbpw = binning.get_n_bands()
        if winv.shape != (4, nbpw, 4, nbpw):
            raise ValueError("Incompatible binning scheme and "
                             "binned MCM.")
        winv = winv.reshape([4*nbpw, 4*nbpw])

    # Read maps
    fields = []
    for fname in fnames:
        mpQ, mpU = hp.read_map(fname, field=[0, 1])
        f = nmt.NmtField(mask, [mpQ, mpU])
        fields.append(f)
    nmaps = len(fields)

    # Compute pseudo-C_\ell
    cls = []
    for icl, i, j in man.cl_pair_iter(nmaps):
        f1 = fields[i]
        f2 = fields[j]
        pcl = binning.bin_cell(nmt.compute_coupled_cell(f1, f2))
        if winv is not None:
            pcl = np.dot(winv, pcl.flatten()).reshape([4, nbpw])
        cls.append(pcl)

    # Save to sacc
    leff = binning.get_effective_ells()
    s = sacc.Sacc()
    for n in names:
        s.add_tracer('Misc', n)
    for icl, i, j in man.cl_pair_iter(nmaps):
        s.add_ell_cl('cl_ee', names[i], names[j], leff, cls[icl][0])
        s.add_ell_cl('cl_eb', names[i], names[j], leff, cls[icl][1])
        if i != j:
            s.add_ell_cl('cl_be', names[i], names[j], leff, cls[icl][2])
        s.add_ell_cl('cl_bb', names[i], names[j], leff, cls[icl][3])
    s.save_fits(fname_out, overwrite=True)


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


def get_noise_cls(meta, lmax, lmin=0, fsky=0.1,
                  is_beam_deconvolved=False):
    """
    Load polarization noise from SO SAT noise model.
    Assume polarization noise is half of that.
    """
    oof_dict = {"pessimistic": 0, "optimistic": 1}

    noise_model = noise_calc.SOSatV3point1(
        survey_years=meta.noise["survey_years"],
        sensitivity_mode=meta.noise["sensitivity_mode"],
        one_over_f_mode=oof_dict[meta.noise["one_over_f_mode"]]
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

    nl_th = {}
    for map_set in meta.map_sets_list:
        freq_tag = meta.freq_tag_from_map_set(map_set)
        if freq_tag in freq_tags:
            nl_th[map_set] = nl_all_frequencies[freq_tag]

    return lth, nl_th


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


def get_beam_windows(meta, plot=False, beam_floor=1.e-2):
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


def create_binning(nside, delta_ell):
    """
    """
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


def m_filter_map(map, map_file, mask, m_cut):
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
                     schedule, thinfp, instrument, band, group_size, nside):
    """
    Applies the TOAST filter to a given map.

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
    schedule : str
        Text file path with the TOAST schedule.
    thinfp : int
        Thinning factor of the number of detectors used in the TOAST
        focalplane.
    instrument : str
        Name of the instrument simulated by TOAST.
    band : str
        Name of the frequency band simulated by TOAST.
    group_size : int
        Group size used for parallelizing filtering with TOAST.
    nside : int
        Healpix Nside parameter of the filtered map.
    """
    import toast
    import sotodlib.toast as sotoast
    from astropy import units as u
    from .toast_utils import (
        apply_scanning, apply_det_pointing_radec, apply_pixels_radec,
        apply_weights_radec, apply_noise_model, apply_scan_map, create_binner,
        apply_demodulation, make_filterbin
    )
    from types import SimpleNamespace
    import toast.mpi

    del map, mask  # delete unused arguments

    comm, procs, rank = toast.mpi.get_world()

    output_dir = map_file.replace('.fits', '/')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize schedule
    schedule_ = toast.schedule.GroundSchedule()

    if not os.path.exists(schedule):
        raise FileNotFoundError(f"The corresponding schedule file {schedule} "
                                "is not stored.")

    # Read schedule
    print('Read schedule')
    schedule_.read(schedule)

    # Setup focal plane
    print('Initialize focal plane and telescope')
    focalplane = sotoast.SOFocalplane(hwfile=None,
                                      telescope=instrument,
                                      sample_rate=40 * u.Hz,
                                      bands=band,
                                      wafer_slots='w25',
                                      tube_slots=None,
                                      thinfp=thinfp,
                                      comm=comm)

    # Setup telescope
    telescope = toast.Telescope(
        name=instrument,
        focalplane=focalplane,
        site=toast.GroundSite("Atacama", schedule_.site_lat,
                              schedule_.site_lon, schedule_.site_alt)
    )

    # Setup toast communicator
    runargs = SimpleNamespace(node_mem=None,
                              group_size=group_size)  # Note the underscore
    group_size = toast.job_group_size(
        comm,
        runargs,
        schedule=schedule_,
        focalplane=focalplane,
    )
    toast_comm = toast.Comm(world=comm,
                            groupsize=group_size)  # Note no underscore

    # Create data object
    data = toast.Data(comm=toast_comm)

    # Apply filters
    print('Apply filters')
    _, sim_gnd = apply_scanning(data, telescope, schedule_)
    data, det_pointing_radec = apply_det_pointing_radec(data, sim_gnd)
    data, pixels_radec = apply_pixels_radec(data, det_pointing_radec, nside)
    data, weights_radec = apply_weights_radec(data, det_pointing_radec)
    data, noise_model = apply_noise_model(data)

    # Scan map
    print('Scan input map')
    data, scan_map = apply_scan_map(data, map_file, pixels_radec,
                                    weights_radec)

    # Create the binner
    binner = create_binner(pixels_radec, det_pointing_radec)

    # Demodulate
    data, weights_radec = apply_demodulation(data, weights_radec, sim_gnd,
                                             binner)

    # Map filterbin
    make_filterbin(data, binner, output_dir)

    if rank == 0:
        # Only one rank can do this
        # Unify name conventions
        if os.path.isfile(output_dir + 'FilterBin_unfiltered_map.fits'):
            # only for TOAST versions < 3.0.0a20
            os.remove(output_dir + 'FilterBin_unfiltered_map.fits')
        os.rename(output_dir + 'FilterBin_filtered_map.fits',
                  output_dir[:-1] + '_filtered.fits')
        if os.path.isdir(output_dir):
            os.rmdir(output_dir)


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


def generate_simulated_maps(meta, id_sim, cl_type, alms_T, alms_E, alms_B,
                            make_plots=False):
    if cl_type == "tf_est":
        for case in ["pureE", "pureB"]:
            if case == "pureE":
                sim = hp.alm2map([alms_T, alms_E, alms_B*0.],
                                 meta.nside, lmax=3*meta.nside - 1)
            elif case == "pureB":
                sim = hp.alm2map([alms_T, alms_E*0., alms_B],
                                 meta.nside, lmax=3*meta.nside - 1)
            map_file = meta.get_map_filename_transfer2(
                id_sim, cl_type, pure_type=case
            )
            hp.write_map(map_file, sim, overwrite=True, dtype=np.float32)

            if make_plots:
                fname = map_file.replace('.fits', '')
                title = map_file.split('/')[-1].replace('.fits', '')
                amp = meta.power_law_pars_tf_est['amp']
                delta_ell = meta.power_law_pars_tf_est['delta_ell']
                pl_index = meta.power_law_pars_tf_est['power_law_index']
                ell0 = 0 if pl_index > 0 else 2 * meta.nside
                var = amp / (ell0 + delta_ell)**pl_index
                plot_map(sim, fname, vrange_T=10*var**0.5,
                         vrange_P=10*var**0.5, title=title,
                         TQU=True)

    else:
        sim = hp.alm2map([alms_T, alms_E, alms_B],
                         meta.nside, lmax=3*meta.nside - 1)
        if not meta.validate_beam:
            map_file = meta.get_map_filename_transfer2(id_sim, cl_type)
            hp.write_map(map_file, sim, overwrite=True, dtype=np.float32)
        else:
            for map_set in meta.map_sets_list:
                _, beam = meta.read_beam(map_set)
                sim_beamed = hp.sphtfunc.smoothing(sim, beam_window=beam)
                map_file = meta.get_map_filename_transfer2(id_sim, cl_type,
                                                           map_set=map_set)
                hp.write_map(map_file, sim_beamed, overwrite=True,
                             dtype=np.float32)

            if make_plots:
                fname = map_file.replace('.fits', '')
                title = map_file.split('/')[-1].replace('.fits', '')
                if cl_type == "tf_val":
                    amp_T = meta.power_law_pars_tf_val['amp']['TT']
                    amp_E = meta.power_law_pars_tf_val['amp']['EE']
                    delta_ell = meta.power_law_pars_tf_val['delta_ell']
                    pl_index = meta.power_law_pars_tf_val['power_law_index']
                    ell0 = 0 if pl_index > 0 else 2 * meta.nside
                    var_T = amp_T / (ell0 + delta_ell)**pl_index
                    var_P = amp_E / (ell0 + delta_ell)**pl_index
                    plot_map(sim, fname, vrange_T=100*var_T**0.5,
                             vrange_P=100*var_P**0.5,
                             title=title, TQU=True)
                elif cl_type == "cosmo":
                    plot_map(sim, fname, title=title, TQU=True)


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


def plot_transfer_function(meta, tf_dict):
    """
    Plot the transfer function given an input dictionary.
    """
    nmt_binning = meta.read_nmt_binning()
    lb = nmt_binning.get_effective_ells()

    fields = ["TT", "TE", "TB", "EE", "EB", "BE", "BB"]
    plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(7, 7, hspace=0.3, wspace=0.3)

    for id1, f1 in enumerate(fields):
        for id2, f2 in enumerate(fields):
            ax = plt.subplot(grid[id1, id2])

            if f1 == "TT" and f2 != "TT":
                ax.axis("off")
                continue
            if f1 in ["TE", "TB"] and f2 not in ["TE", "TB"]:
                ax.axis("off")
                continue
            if f1 in ["EE", "EB", "BE", "BB"] \
                    and f2 not in ["EE", "EB", "BE", "BB"]:
                ax.axis("off")
                continue

            ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)

            ax.errorbar(
                lb, tf_dict[f1, f2], tf_dict[f1, f2, "std"],
                marker=".", markerfacecolor="white",
                color="navy")

            if not ([id1, id2] in [[0, 0], [2, 1],
                                   [2, 2], [6, 3],
                                   [6, 4], [6, 5],
                                   [6, 6]]):
                ax.set_xticks([])
            else:
                ax.set_xlabel(r"$\ell$", fontsize=14)

            if f1 == f2:
                ax.axhline(1., color="k", ls="--")
            else:
                ax.axhline(0, color="k", ls="--")

            ax.set_xlim(meta.lmin, meta.lmax)

    plot_dir = meta.plot_dir_from_output_dir(meta.coupling_directory)
    plt.savefig(f"{plot_dir}/transfer.pdf", bbox_inches="tight")


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


class PipelineManager(object):
    def __init__(self, fname_config):
        with open(fname_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.nside = self.config['nside']
        self.fname_mask = self.config['mask']
        self.fname_binary_mask = self.config['binary_mask']
        self.bpw_edges = None
        self.pl_names = np.loadtxt(self.config['pl_sims'], dtype=str)
        self.val_names = np.loadtxt(self.config['val_sims'], dtype=str)
        self.pl_input_dir = self.config['pl_sims_dir']
        self.val_input_dir = self.config['val_sims_dir']
        self._get_cls_PL()
        self._get_cls_val()

        self.stname_mcm = 'mcm'
        self.stname_filtpl = 'filter_PL'
        self.stname_pclpl_in = 'pcl_PL_in'
        self.stname_pclpl_filt = 'pcl_PL_filt'
        self.stname_filtval = 'filter_val'
        self.stname_pclval_in = 'pcl_val_in'
        self.stname_pclval_filt = 'pcl_val_filt'
        self.stname_clval = 'cl_val'
        self.stname_transfer = 'transfer'

    def get_filename(self, product, out_base_dir, simname=None):
        if product == 'mcm':  # NaMaster's MCM
            fname = os.path.join(out_base_dir, '..',
                                 self.stname_mcm, 'mcm.npz')
        if product == 'mcm_plots':  # NaMaster's MCM plots dir
            fname = os.path.join(out_base_dir, '..',
                                 self.stname_mcm)
        if product == 'pl_sim_input':  # Input PL sims
            fname = os.path.join(self.pl_input_dir,
                                 simname+'.fits')
        if product == 'pl_sim_filtered':  # Filtered PL sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_filtpl,
                simname+'.fits')
        if product == 'pcl_pl_sim_input':  # PCL of input PL sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclpl_in,
                simname+'_pcl_in.fits')
        if product == 'pcl_pl_sim_filtered':  # PCL of filtered PL sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclpl_filt,
                simname+'_pcl_filt.fits')
        if product == 'val_sim_input':  # Input validation sims
            fname = os.path.join(self.val_input_dir,
                                 simname+'.fits')
        if product == 'val_sim_filtered':  # Filtered validation sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_filtval,
                simname+'.fits')
        if product == 'pcl_val_sim_input':  # PCL of input validations sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclval_in,
                simname+'_pcl_in.fits')
        if product == 'pcl_val_sim_filtered':
            # PCL of filtered validation sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclval_filt,
                simname+'_pcl_filt.fits')
        if product == 'cl_val_sim':  # CL of filtered validation sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_clval,
                simname+'_cl.fits')
        if product == 'transfer_function':
            fname = os.path.join(
                out_base_dir, '..', self.stname_transfer, 'transfer.npz')
        return fname

    def _get_cls_PL(self):
        d = np.load(self.config['cl_PL'])
        lin = d['ls']
        ls = np.arange(3*self.nside)
        self.cls_PL = []
        for kind in ['EE', 'EB', 'BE', 'BB']:
            cli = interp1d(lin, d[f'cl{kind}'], bounds_error=False,
                           fill_value=0)
            self.cls_PL.append(cli(ls))
        self.cls_PL = np.array(self.cls_PL)

    def _get_cls_val(self):
        d = np.load(self.config['cl_val'])
        lin = d['ls']
        ls = np.arange(3*self.nside)
        self.cls_val = []
        for kind in ['EE', 'EB', 'BE', 'BB']:
            cli = interp1d(lin, d[f'cl{kind}'], bounds_error=False,
                           fill_value=0)
            self.cls_val.append(cli(ls))
        self.cls_val = np.array(self.cls_val)

    def get_bpw_edges(self):
        if self.bpw_edges is None:
            self.bpw_edges = np.load(self.config['bpw_edges'])['bpw_edges']
        return self.bpw_edges

    def get_nmt_bins(self):
        import pymaster as nmt
        bpw_edges = self.get_bpw_edges()
        b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
        return b

    def cl_pair_iter(self, nmaps):
        icl = 0
        for i in range(nmaps):
            for j in range(i, nmaps):
                yield icl, i, j
                icl += 1

    def val_sim_names(self, sim0, nsims, output_dir, which='input'):
        if nsims == -1:
            names = self.val_names[sim0:]
        else:
            names = self.val_names[sim0:sim0+nsims]
        fnames = []
        for n in names:
            if which == 'names':
                fn = n
            elif which == 'input':
                fn = self.get_filename('val_sim_input',
                                       output_dir, n)
            elif which in ['filtered', 'decoupled']:
                fn = self.get_filename('val_sim_filtered',
                                       output_dir, n)
            elif which == 'input_Cl':
                fn = self.get_filename("pcl_val_sim_input",
                                       output_dir, n)
            elif which == 'filtered_Cl':
                fn = self.get_filename("pcl_val_sim_filtered",
                                       output_dir, n)
            elif which == 'decoupled_Cl':
                fn = self.get_filename("cl_val_sim",
                                       output_dir, n)
            else:
                raise ValueError(f"Unknown kind {which}")
            fnames.append(fn)
        return fnames

    def pl_sim_names_EandB(self, sim0, nsims, output_dir, which):
        return self.pl_sim_names(sim0, nsims, output_dir,
                                 which=which, EandB=True)

    def pl_sim_names(self, sim0, nsims, output_dir, which='input',
                     EandB=False):
        if nsims == -1:
            names = self.pl_names[sim0:]
        else:
            names = self.pl_names[sim0:sim0+nsims]
        fnames = []
        for n in names:
            if which == 'names':
                fE = n+'_E'
                fB = n+'_B'
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'input':
                fE = self.get_filename('pl_sim_input',
                                       output_dir, n+'_E')
                fB = self.get_filename('pl_sim_input',
                                       output_dir, n+'_B')
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'filtered':
                fE = self.get_filename('pl_sim_filtered',
                                       output_dir, n+'_E')
                fB = self.get_filename('pl_sim_filtered',
                                       output_dir, n+'_B')
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'input_Cl':
                fn = self.get_filename('pcl_pl_sim_input', output_dir, n)
                fnames.append(fn)
            elif which == 'filtered_Cl':
                fn = self.get_filename('pcl_pl_sim_filtered', output_dir, n)
                fnames.append(fn)
            else:
                raise ValueError(f"Unknown kind {which}")
        return fnames


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
