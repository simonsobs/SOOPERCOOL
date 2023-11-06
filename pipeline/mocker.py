import argparse
import healpy as hp
from bbmaster.utils import *
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_2 as noise_calc
import yaml




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    o = parser.parse_args()

    #man = PipelineManager(o.globals)
    with open(o.globals, "r") as f:
        global_dict = yaml.safe_load(f)

    map_dir = global_dict["map_directory"]
    mask_dir = global_dict["masks"]["mask_directory"]
    beam_dir = global_dict["beam_directory"]

    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(beam_dir, exist_ok=True)

    cosmo = global_dict["sim_pars"]["cosmology"]

    lth, psth = theory_cls(
        cosmo,
        lmax=global_dict["lmax"]
    )

    # Save binary mask
    binary_mask = hp.read_map(f"{mask_dir}/{global_dict['masks']['binary_mask']}")
    fsky = np.mean(binary_mask)

    # Load noise curves
    noise_model = noise_calc.SOSatV3point1(sensitivity_mode='baseline')
    lth, nlth_T, nlth_P = noise_model.get_noise_curves(
        fsky,
        global_dict["lmax"],
        delta_ell=1,
        deconv_beam=False
    )
    # Only support polarization noise at the moment
    nlth_dict = {
        "T": None,
        "P": {freq_band: nlth_P[i] for i, freq_band in enumerate(noise_model.get_bands())}
    }

    # Load hitmap
    hitmap = hp.read_map(global_dict["sim_pars"]["hitmap_file"])
    hitmap = hp.ud_grade(hitmap, global_dict["nside"], power=-2)

    # Load and save beams
    beam_arcmin = {
        freq_band: beam_arcmin for freq_band, beam_arcmin in zip(noise_model.get_bands(), noise_model.get_beams())
    }
    beams = {}
    for tag, tag_meta in global_dict["map_sets"].items():
        beams[tag] = beam_gaussian(lth, beam_arcmin[tag_meta["freq_tag"]])
        # Save beams
        np.savetxt(f"{beam_dir}/beam_{tag_meta['file_root']}.dat", np.transpose([lth, beams[tag]]))
    
    hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]
    cmb_map = hp.synfast([psth[k] for k in hp_ordering], global_dict["nside"])
    for tag, tag_meta in global_dict["map_sets"].items():

        freq_tag = tag_meta["freq_tag"]
        cmb_map_beamed = hp.sphtfunc.smoothing(cmb_map, fwhm=np.deg2rad(beam_arcmin[freq_tag] / 60))

        for id_split in range(tag_meta["n_splits"]):
            noise_map = generate_noise_map(nlth_dict["T"], nlth_dict["P"][tag_meta["freq_tag"]], hitmap, tag_meta["n_splits"])
            split_map = cmb_map_beamed + noise_map

            split_map *= binary_mask

            map_file_name = f"{tag_meta['file_root']}_split_{id_split}.fits"
            hp.write_map(
                f"{map_dir}/{map_file_name}",
                split_map,
                overwrite=True            
            )
                
            for i, m in enumerate("TQU"):
                vrange = 300 if m == "T" else 6
                plt.figure(figsize=(16,9))
                hp.mollview(split_map[i], title=tag, cmap=cm.coolwarm, min=-vrange, max=vrange)
                hp.graticule()
                plt.savefig(f"{map_dir}/{m}_{map_file_name.replace('fits', 'png')}")