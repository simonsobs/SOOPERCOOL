from sotodlib.core import Context
from sotodlib.core import metadata
from sotodlib.tod_ops import (flags, fft_ops, filters,
                              detrend_tod, apodize, sub_polyf)
from sotodlib.hwp import hwp
import sotodlib.coords.demod as demod_mm

from pixell import enmap, enplot, reproject, utils
import so3g
import pandas as pd
import sqlite3 as sq

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/global/homes/k/kwolz/bbdev/SOOPERCOOL/soopercool")
from utils import power_law_cl


def model_func(x, sigma, fk, alpha):
    """
    """
    return sigma**2 * (1 + (x/fk)**alpha)


def log_fit_func(x, sigma, fk, alpha):
    """
    """
    return np.log(model_func(x, sigma, fk, alpha))


def generate_gaussian_map(ps_dict, map_shape, wcs, seed=1234):
    """
    ps_dict has elements "TT", "TE", "TB", "EE", "EB", "BB"
    """
    cov_ind = {
        "TT": (0,0), "TE": (0,1), "TB": (0,2),
        "EE": (1,1), "EB": (1,2), "BB": (2,2)
    }
    ells = np.arange(0, len(ps_dict["EE"]), 1)

    # Gaussian power spectrum covariance
    cov = np.zeros([3, 3, len(ps_dict["EE"])])
    for field_pair, ps in ps_dict.items():
        cov[cov_ind[field_pair]][2:] = 2*ps[2:]/(2*ells[2:] + 1)
    cov_upper = cov
    for il in range(len(ps_dict["EE"])):
        cov_upper = cov[:, :, il]
        cov[:, :, il] = cov_upper + cov_upper.T - np.diag(cov_upper.diagonal())
    
    # plt.clf()
    # plt.plot(ells, cov[2,2,:], label="BB")
    # plt.plot(ells, cov[1,1,:], label="EE")
    # plt.plot(ells, cov[1,2,:], label="EB")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    # plt.show()

    return enmap.rand_map((3, map_shape[0], map_shape[1]),
                           wcs, cov, spin=[0, 2], seed=seed)


def get_aman(context, obs_id, wafer, freq, thinning_factor=1.0, seed=1234):
    ctx = Context(context)
    if (thinning_factor > 1) or (thinning_factor <= 0):
        raise ValueError(f"Thinning: {thinning_factor} is not between 0 and 1")
    print(f"Doing {obs_id} / {wafer} / {freq}")
    # Get metadata for this observation
    meta = ctx.get_meta(obs_id)
    # Apply detector cuts
    print("All", meta.dets.count)
    # Select wafer and frequency (remember atomic maps are saved per wafer and per frequency)
    meta.restrict('dets', meta.det_info.wafer_slot == wafer)
    print(wafer, meta.dets.count)
    meta.restrict('dets', meta.det_info.wafer.bandpass == freq)
    print(freq, meta.dets.count)
    # Remove detectors with rubbish coordinates
    meta.restrict('dets', meta.dets.vals[~np.isnan(meta.focal_plane.xi)])
    print("xi", meta.dets.count)
    meta.restrict('dets', meta.dets.vals[~np.isnan(meta.focal_plane.gamma)])
    print("gamma", meta.dets.count)
    if thinning_factor < 1:
        np.random.seed(seed)
        keep = np.random.rand(meta.dets.count) <= thinning_factor
        meta.restrict('dets', meta.dets.vals[keep])
        print("thinning", meta.dets.count)    
    print("Getting pointing information")
    aman = ctx.get_obs(meta, no_signal=True)
    # aman now contains the tod pointing information (with no signal)
    return aman

def get_obs_id_from_map_id(context, map_id):
    ctx = Context(context)
    # Query all observations
    obslist = ctx.obsdb.query()
    # Extract their ctimes
    ctimes_tod = np.array([int(ob['obs_id'].split('_')[1]) for ob in obslist])
    # Find the time-closest observation for each atomic map and get its obs_id
    #obs_ids = {}
    #for t in map_ids:
    ix = np.argmin(np.fabs(int(map_id)-ctimes_tod))
    t_tod = ctimes_tod[ix]
    tdiff = np.fabs(int(map_id)-t_tod)
    assert tdiff <= 5
    #obs_ids[map_ids] = obslist['obs_id'][ix]
    return obslist['obs_id'][ix]

def filter_sim(aman, input_map, wcs, return_nofilter=False):
    """
    """
    # Observe map into TOD
    print("Observing into TOD")
    dsT_sim, demodQ_sim, demodU_sim = demod_mm.from_map(
        aman, input_map, wrap=True
    )

    if return_nofilter:
        # Map it back before filtering
        print("Mapping back")
        res_nofilt = demod_mm.make_map(aman, wcs_kernel=wcs)

    #Process the TOD (here's where filtering happens)
    print("Calibrating TOD")
    aman = calibrate_obs_tomoki(aman)

    # Map it again
    print("Mapping back")
    res_filt = demod_mm.make_map(aman, wcs_kernel=wcs)

    if return_nofilter:
        return (res_filt['weighted_map'], res_filt['weight'],
                res_nofilt['weighted_map'], res_nofilt['weight'])

    return res_filt['weighted_map'], res_filt['weight']


def get_CAR_template(ncomp, res, dec_cut=[-75, 25]):
    """
    """
    if dec_cut is None:
        shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60))
    else:
        shape, wcs = enmap.band_geometry(np.deg2rad(dec_cut), res=np.deg2rad(res/60))
    shape = (ncomp, *shape)
    return enmap.zeros(shape, wcs)


def get_CAR_template(ncomp, res, dec_cut=None):
    """
    """
    if dec_cut is None:
        shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60), proj="car")
    else:
        shape, wcs = enmap.band_geometry(np.deg2rad(dec_cut), res=np.deg2rad(res/60))
    shape = (ncomp, *shape)
    return enmap.zeros(shape, wcs)


def calibrate_obs_tomoki(obs, dtype_tod=np.float32, site='so_sat1',
                         det_left_right=False, det_in_out=False,
                         det_upper_lower=False):
    """
    """
    from scipy.optimize import curve_fit
    from scipy.stats import kurtosis, skew

    obs.wrap("weather", np.full(1, "toco"))
    obs.wrap("site",    np.full(1, site))
    obs.flags.wrap(
        'glitch_flags',
        so3g.proj.RangesMatrix.zeros(obs.shape[:2]),
        [(0, 'dets'), (1, 'samps')]
    )
    # Restrict non optical detectors, which have nans in their
    # focal plane coordinates and will crash the mapmaking operation.
    obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
    obs.restrict(
        'dets',
        obs.dets.vals[(0.2<obs.det_cal.r_frac)&(obs.det_cal.r_frac<0.8)]
    )
    
    if obs.signal is not None:
        if det_left_right or det_in_out or det_upper_lower:
            # we add a metadata manager for the detector flags
            obs.wrap('det_flags', metadata.for_tod(obs))
            if det_left_right or det_in_out:
                xi = obs.focal_plane.xi
                # sort xi 
                xi_median = np.median(xi)    
            if det_upper_lower or det_in_out:
                eta = obs.focal_plane.eta
                # sort eta
                eta_median = np.median(eta)
            if det_left_right:
                mask = xi <= xi_median
                obs.det_flags.wrap_dets('det_left', np.logical_not(mask))
                mask = xi > xi_median
                obs.det_flags.wrap_dets('det_right', np.logical_not(mask))
            if det_upper_lower:
                mask = eta <= eta_median
                obs.det_flags.wrap_dets('det_lower', np.logical_not(mask))
                mask = eta > eta_median
                obs.det_flags.wrap_dets('det_upper', np.logical_not(mask))
            if det_in_out:
                # the bounding box is the center of the detset
                xi_center = np.min(xi) + 0.5 * (np.max(xi) - np.min(xi))
                eta_center = np.min(eta) + 0.5 * (np.max(eta) - np.min(eta))
                radii = np.sqrt((xi_center-xi)**2 + (eta_center-eta)**2)
                radius_median = np.median(radii)
                mask = radii <= radius_median
                obs.det_flags.wrap_dets('det_in', np.logical_not(mask))
                mask = radii > radius_median
                obs.det_flags.wrap_dets('det_out', np.logical_not(mask))
        
        nperseg = 200*1000
        
        obs.focal_plane.gamma = np.arctan(np.tan(obs.focal_plane.gamma))
        flags.get_turnaround_flags(obs, t_buffer=0.1, truncate=True)
        obs.signal = np.multiply(obs.signal.T, obs.det_cal.phase_to_pW).T
        freq, Pxx = fft_ops.calc_psd(obs, nperseg=nperseg, merge=True)
        #print('First psd, number of nusamps', obs.nusamps.count)
        #print('First psd, shape of P',Pxx.shape)
        wn = fft_ops.calc_wn(obs)
        obs.wrap('wn', wn, [(0, 'dets')])

        # satp3
        bgs = obs.det_cal.bg
        bgs90 = [0,1,4,5,8,9]
        bgs150 = [2,3,6,7,10,11]
        m90 = np.array([bg in bgs90 for bg in bgs])
        m150 = np.array([bg in bgs150 for bg in bgs])
        #obs = obs.restrict('dets', obs.dets.vals[m90]) # SA TODO
        if obs.hwp_solution.primary_encoder == 1:
            if np.all(obs.det_cal.bg.all() in bgs90):
                # 90 GHz
                print('90')
                obs.hwp_angle = np.mod(
                    -1*np.unwrap(obs.hwp_angle) + np.deg2rad(-1.66-2.29+90),
                    2*np.pi
                )
            elif np.all(obs.det_cal.bg.all() in bgs150):
                # 150 GHz
                print('150')
                obs.hwp_angle = np.mod(
                    -1*np.unwrap(obs.hwp_angle) + np.deg2rad(-1.66-1.99+90),
                    2*np.pi
                )
        elif obs.hwp_solution.primary_encoder == 2:
            print('2')
            if np.all(obs.det_cal.bg.all() in bgs90):
                # 90 GHz
                obs.hwp_angle = np.mod(
                    -1*np.unwrap(obs.hwp_angle) + np.deg2rad(-1.66-2.29-90),
                    2*np.pi
                )
            elif np.all(obs.det_cal.bg.all() in bgs150):
                # 150 GHz
                obs.hwp_angle = np.mod(
                    -1*np.unwrap(obs.hwp_angle) + np.deg2rad(-1.66-1.99-90),
                    2*np.pi
                )

        obs.restrict('dets', obs.dets.vals[(20<obs.wn*1e6)&(obs.wn*1e6<40)])
        print(f'dets: {obs.dets.count}')
        
        # peak to peak restrict
        obs.restrict('dets', obs.dets.vals[np.ptp(obs.signal, axis=1) < 0.5])
        print(f'dets: {obs.dets.count}')
        
        if obs.dets.count<=1: return obs
        
        hwp.get_hwpss(obs)
        hwp.subtract_hwpss(obs)
        obs.move('signal', None)
        obs.move('hwpss_remove', 'signal')
        freq, Pxx = fft_ops.calc_psd(obs, nperseg=nperseg, merge=False)
        #print('Second psd, number of nusamps', obs.nusamps.count)
        #print('Second psd, shape of P',Pxx.shape)
        obs.Pxx = Pxx
        
        detrend_tod(obs, method='median')
        apodize.apodize_cosine(obs, apodize_samps=2000)
        
        # the demodulation will happen here
        speed = (np.sum(np.abs(np.diff(np.unwrap(obs.hwp_angle)))) /
                (obs.timestamps[-1] - obs.timestamps[0])) / (2 * np.pi)
        bpf_center = 4 * speed
        bpf_width = speed * 2. * 0.9
        bpf_cfg = {'type': 'sine2',
                   'center': bpf_center,
                   'width': bpf_width,
                   'trans_width': 0.1}

        lpf_cutoff = speed * 0.9
        lpf_cfg = {'type': 'sine2',
                   'cutoff': lpf_cutoff,
                   'trans_width': 0.1}
        hwp.demod_tod(obs, bpf_cfg=bpf_cfg, lpf_cfg=lpf_cfg)
        
        obs.restrict(
            'samps',
            (obs.samps.offset+2000, obs.samps.offset + obs.samps.count-2000)
        )
        obs.move('signal', None)
        detrend_tod(obs, signal_name='dsT', method='linear')
        detrend_tod(obs, signal_name='demodQ', method='linear')
        detrend_tod(obs, signal_name='demodU', method='linear')
        freq, Pxx_demodQ = fft_ops.calc_psd(obs, signal=obs.demodQ,
                                            nperseg=nperseg, merge=False)
        freq, Pxx_demodU = fft_ops.calc_psd(obs, signal=obs.demodU,
                                            nperseg=nperseg, merge=False)
        obs.wrap('Pxx_demodQ', Pxx_demodQ, [(0, 'dets'), (1, 'nusamps')])
        obs.wrap('Pxx_demodU', Pxx_demodU, [(0, 'dets'), (1, 'nusamps')])
        
        mask = np.ones_like(obs.dsT, dtype='bool')
    
        lamQ, lamU = [], []
        AQ, AU = [], []

        for di, det in enumerate(obs.dets.vals[:]):
            x = obs.dsT[di][mask[di]]
            y1 = obs.demodQ[di][mask[di]]
            y2 = obs.demodU[di][mask[di]]

            z1 = np.polyfit(x, y1, 1)
            z2 = np.polyfit(x, y2, 1)
            _lamQ, _AQ = z1[0], z1[1]
            _lamU, _AU = z2[0], z2[1]

            lamQ.append(_lamQ)
            lamU.append(_lamU)
            AQ.append(_AQ)
            AU.append(_AU)

        lamQ, lamU = np.array(lamQ), np.array(lamU)
        obs.wrap('lamQ', lamQ, [(0, 'dets')])
        obs.wrap('lamU', lamU, [(0, 'dets')])

        AQ, AU = np.array(AQ), np.array(AU)
        obs.wrap('AQ', AQ, [(0, 'dets')])
        obs.wrap('AU', AU, [(0, 'dets')])

        obs.demodQ -= (obs.dsT * obs.lamQ[:, np.newaxis]
                       + obs.AQ[:, np.newaxis])
        obs.demodU -= (obs.dsT * obs.lamU[:, np.newaxis]
                       + obs.AU[:, np.newaxis])

        freq, Pxx_demodQ_new = fft_ops.calc_psd(obs, signal=obs.demodQ,
                                                nperseg=nperseg, merge=False)
        freq, Pxx_demodU_new = fft_ops.calc_psd(obs, signal=obs.demodU,
                                                nperseg=nperseg, merge=False)
        obs.Pxx_demodQ = Pxx_demodQ_new
        obs.Pxx_demodU = Pxx_demodU_new
        
        mask_valid_freqs = (1e-4<obs.freqs) & (obs.freqs < 1.9)
        x = obs.freqs[mask_valid_freqs]
        obs.wrap_new('sigma', ('dets', ))
        obs.wrap_new('fk', ('dets', ))
        obs.wrap_new('alpha', ('dets', ))

        for di, det in enumerate(obs.dets.vals):
            y = obs.Pxx_demodQ[di, mask_valid_freqs]
            popt, pcov = curve_fit(
                log_fit_func, x, np.log(y),
                p0=(np.sqrt(np.median(y[x>0.2])), 0.01, -2.), maxfev=100000
            )
            obs.sigma[di] = popt[0]
            obs.fk[di] = popt[1]
            obs.alpha[di] = popt[2]
        
        kurt_threshold=0.5
        skew_threshold=0.5
        
        valid_scan = np.logical_and(
            np.logical_or(obs.flags["left_scan"].mask(),
                          obs.flags["right_scan"].mask()),
                          ~obs.flags["turnarounds"].mask())

        subscan_indices_l = sub_polyf._get_subscan_range_index(
            obs.flags["left_scan"].mask()
        )
        subscan_indices_r = sub_polyf._get_subscan_range_index(
            obs.flags["right_scan"].mask()
        )
        subscan_indices = np.vstack([subscan_indices_l, subscan_indices_r])
        subscan_indices= subscan_indices[np.argsort(subscan_indices[:, 0])]

        subscan_Qstds = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Ustds = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Qkurt = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Ukurt = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Qskew = np.zeros([obs.dets.count, len(subscan_indices)])
        subscan_Uskew = np.zeros([obs.dets.count, len(subscan_indices)])

        for subscan_i, subscan in enumerate(subscan_indices):
            _Qsig= obs.demodQ[:,subscan[0]:subscan[1]+1]
            _Usig= obs.demodU[:,subscan[0]:subscan[1]+1]

            _Qmean = np.mean(_Qsig, axis=1)[:,np.newaxis]
            _Umean = np.mean(_Usig, axis=1)[:,np.newaxis]

            _Qstd = np.std(_Qsig, axis=1)
            _Ustd = np.std(_Usig, axis=1)

            _Qkurt = kurtosis(_Qsig, axis=1)
            _Ukurt = kurtosis(_Usig, axis=1)

            _Qskew = skew(_Qsig, axis=1)
            _Uskew = skew(_Usig, axis=1)

            obs.demodQ[:,subscan[0]:subscan[1]+1] -= _Qmean
            obs.demodU[:,subscan[0]:subscan[1]+1] -= _Umean

            subscan_Qstds[:, subscan_i] = _Qstd
            subscan_Ustds[:, subscan_i] = _Ustd
            subscan_Qkurt[:, subscan_i] = _Qkurt
            subscan_Ukurt[:, subscan_i] = _Ukurt
            subscan_Qskew[:, subscan_i] = _Qskew
            subscan_Uskew[:, subscan_i] = _Uskew

        badsubscan_indicator = ((np.abs(subscan_Qkurt) > kurt_threshold)
                                | (np.abs(subscan_Ukurt) > kurt_threshold)
                                | (np.abs(subscan_Qskew) > skew_threshold)
                                | (np.abs(subscan_Uskew) > skew_threshold))
        badsubscan_flags = np.zeros([obs.dets.count, obs.samps.count],
                                    dtype='bool')

        for subscan_i, subscan in enumerate(subscan_indices):
            flag_values = badsubscan_indicator[:, subscan_i, np.newaxis]
            badsubscan_flags[:, subscan[0]:subscan[1]+1] = flag_values
        badsubscan_flags = so3g.proj.RangesMatrix.from_mask(badsubscan_flags)

        obs.flags.wrap('bad_subscan', badsubscan_flags)
        
        filt = filters.counter_1_over_f(np.median(obs.fk),
                                        -2*np.median(obs.alpha))
        obs.demodQ = filters.fourier_filter(obs, filt, signal_name='demodQ')
        obs.demodU = filters.fourier_filter(obs, filt, signal_name='demodU')

        freq, Pxx_demodQ = fft_ops.calc_psd(obs, signal=obs.demodQ,
                                            nperseg=nperseg, merge=False)
        freq, Pxx_demodU = fft_ops.calc_psd(obs, signal=obs.demodU,
                                            nperseg=nperseg, merge=False)

        obs.Pxx_demodQ = Pxx_demodQ
        obs.Pxx_demodU = Pxx_demodU
        
        wn = fft_ops.calc_wn(obs, obs.Pxx_demodQ, low_f=0.1, high_f=1.)
        obs.wrap('inv_var', wn**(-2), [(0, 'dets')])
        if True:
            lo, hi = np.percentile(obs.inv_var, [3, 97])
            obs.restrict(
                'dets',
                obs.dets.vals[(lo < obs.inv_var) & (obs.inv_var < hi)]
            )
        if obs.dets.count<=1:
            return obs
    
        glitches_T = flags.get_glitch_flags(
            obs, signal_name='dsT', merge=True, name='glitches_T'
        )
        glitches_Q = flags.get_glitch_flags(
            obs, signal_name='demodQ', merge=True, name='glitches_Q'
        )
        glitches_U = flags.get_glitch_flags(
            obs, signal_name='demodU', merge=True, name='glitches_U'
        )
        obs.flags.reduce(
            flags=['glitches_T', 'glitches_Q', 'glitches_U'],
            method='union', wrap=True, new_flag='glitches', remove_reduced=True
        )
        obs.flags.move('glitch_flags', None)
        obs.flags.reduce(
            flags=['turnarounds', 'bad_subscan', 'glitches'],
            method='union', wrap=True, new_flag='glitch_flags',
            remove_reduced=True
        )
    return obs
