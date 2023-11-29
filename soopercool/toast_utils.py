import numpy as np
import warnings
import yaml
import toast
import toast.ops
import pixell.fft
import subprocess
from sotodlib import core
from sotodlib.site_pipeline.preprocess_tod import _build_pipe_from_configs
pixell.fft.engine = "fftw"
warnings.simplefilter("ignore")


"""
A collection of useful functions written by SA, and/or other SO members.
"""


def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        print("Command output:", result.stdout)
        print("Command error (if any):", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error running command:", e)


def apply_scanning(data, telescope, schedule):
    sim_gnd = toast.ops.SimGround(
        telescope=telescope,
        schedule=schedule,
        weather="atacama",
        hwp_angle='hwp_angle',
        hwp_rpm=120,
    )
    sim_gnd.apply(data)
    return data, sim_gnd


def apply_det_pointing_radec(data, sim_gnd):
    det_pointing_radec = toast.ops.PointingDetectorSimple(
        name='det_pointing_radec',
        quats='quats_radex',
        shared_flags=None,
    )
    det_pointing_radec.boresight = sim_gnd.boresight_radec
    det_pointing_radec.apply(data)
    return data, det_pointing_radec


def apply_det_pointing_azel(data, sim_gnd):
    det_pointing_azel = toast.ops.PointingDetectorSimple(
        name='det_pointing_azel',
        quats='quats_azel',
        shared_flags=None,
    )
    det_pointing_azel.boresight = sim_gnd.boresight_azel
    det_pointing_azel.apply(data)
    return data, det_pointing_azel


def apply_pixels_radec(data, det_pointing_radec, nside):
    pixels_radec = toast.ops.pixels_healpix.PixelsHealpix(
        name="pixels_radec",
        pixels='pixels',
        nside=nside,
        nside_submap=8,
    )
    pixels_radec.detector_pointing = det_pointing_radec
    pixels_radec.apply(data)
    return data, pixels_radec


def apply_weights_radec(data, det_pointing_radec):
    weights_radec = toast.ops.stokes_weights.StokesWeights(
        mode="IQU",  # The Stokes weights to generate (I or IQU)
        name="weights_radec",  # The 'name' of this class instance,
        hwp_angle="hwp_angle",
    )
    weights_radec.detector_pointing = det_pointing_radec
    weights_radec.apply(data)
    return data, weights_radec


def apply_scan_map(data, file, pixels_radec, weights_radec):
    scan_map = toast.ops.ScanHealpixMap(
        name='scan_map',
        file=file,
    )
    scan_map.pixel_pointing = pixels_radec
    scan_map.stokes_weights = weights_radec
    scan_map.apply(data)
    return data, scan_map


def apply_noise_model(data):
    noise_model = toast.ops.DefaultNoiseModel(
        name='default_model',
        noise_model='noise_model',
    )
    noise_model.apply(data)
    return data, noise_model


def apply_sim_noise(data):
    sim_noise = toast.ops.SimNoise()
    sim_noise.apply(data)
    return data, sim_noise


def create_binner(pixels_radec, det_pointing_radec):
    binner = toast.ops.BinMap()
    binner.pixel_pointing = pixels_radec
    binner.pixel_pointing.detector_pointing = det_pointing_radec
    return binner


def apply_demodulation(data, weights_radec, sim_gnd, binner):
    demod = toast.ops.Demodulate()
    demod.stokes_weights = weights_radec
    demod.hwp_angle = sim_gnd.hwp_angle
    data = demod.apply(data)
    demod_weights = toast.ops.StokesWeightsDemod()
    binner.stokes_weights = demod_weights
    return data, demod_weights


def make_filterbin(data, binner, output_dir):
    filterbin = toast.ops.FilterBin()
    filterbin.binning = binner
    filterbin.output_dir = output_dir
    filterbin.write_hits = False
    filterbin.write_cov = False
    filterbin.write_rcond = False
    filterbin.apply(data)


def preprocess(aman):
    # Let's load the config file created
    configs = "./preprocess/pipe_s0002_sim_preprocess.yaml"
    configs = yaml.safe_load(open(configs, "r"))
    # And build the pipeline including the filters
    # specified in the config
    pipe = _build_pipe_from_configs(configs)
    proc_aman = core.AxisManager(aman.dets, aman.samps)
    for pi in np.arange(3):
        # 1) Detrended
        # 2) Apodize
        # 3) Demodulate
        process = pipe[pi]
        # print(f"Processing {process.name}")
        process.process(aman, proc_aman)
        process.calc_and_save(aman, proc_aman)
    return aman, proc_aman
