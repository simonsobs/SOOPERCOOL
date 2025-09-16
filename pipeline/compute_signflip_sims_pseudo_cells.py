from soopercool import BBmeta
from soopercool import ps_utils as pu
from soopercool import map_utils as mu
from pixell import enmap
from soopercool import mpi_utils as mpi
import argparse
import numpy as np
import pymaster as nmt
import os


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    # Preferred template priority:
    # 1) CLI --car-template
    # 2) meta.car_template (if present)
    car_template = args.car_template or getattr(meta, "car_template", None)

    out_dir = meta.output_directory
    external_dir = meta.external_dir
    signflip_dir = meta.signflip_dir
    cells_dir = f"{out_dir}/cells_signflip/"
    couplings_dir = f"{out_dir}/couplings"
    sims_dir = f"{signflip_dir}/"

    BBmeta.make_dir(cells_dir)

    #mask = mu.read_map(meta.masks["analysis_mask"],
    #                   pix_type=meta.pix_type)
    # Defer reading mask until we know a valid template (first map if needed)
    mask = None

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    n_bins = nmt_bins.get_n_bands()

    inv_couplings_beamed = {}

    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):
        inv_couplings_beamed[ms1, ms2] = np.load(f"{couplings_dir}/couplings_{ms1}_{ms2}.npz")["inv_coupling"].reshape([n_bins*9, n_bins*9]) # noqa

    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):
        print(f"Sim {id_sim}")
        base_dir = f"{sims_dir}/"

        # Create namaster fields
        fields = {}
        for map_name in meta.maps_list:
            
            map_set, id_bundle = map_name.split("__")
            sat_parts = map_set.split('_')
            sat = '_'.join([part.lower() for part in sat_parts if part.startswith('SAT')])
            non_sat_parts = [p for p in sat_parts if not p.startswith('SAT')]
            frq = non_sat_parts[0] if len(non_sat_parts) > 0 else None
            area = non_sat_parts[1] if len(non_sat_parts) > 1 else None
            ssplit = '_'.join(non_sat_parts[2:]) if len(non_sat_parts) > 2 else ''

            map_fname = f"{base_dir}{sat}_{frq}_{ssplit}_bundle{id_bundle}_{id_sim:04d}_map.fits"
            print(map_fname)
            
            # Choose a usable template for this run:
            # prefer --car-template or meta.car_template; otherwise use the first map file
            template_for_this_run = car_template or map_fname

            m = mu.read_map(
                map_fname,
                pix_type=meta.pix_type,
                car_template=template_for_this_run,
                convert_K_to_muK=True
            )

            # Read mask now that we have a template
            if mask is None:
                mask = mu.read_map(
                    meta.masks["analysis_mask"],
                    pix_type=meta.pix_type,
                    car_template=template_for_this_run
                )

            wcs = None
            # Align to a common CAR footprint if available
            wcs = getattr(m, "wcs", None)
            if wcs is not None:
                # If we have a declared template, use it; otherwise use map's own geometry
                if car_template is not None:
                    tshape, twcs = enmap.read_map_geometry(car_template)
                else:
                    tshape, twcs = m.shape, m.wcs
                shape, wcs = enmap.overlap(m.shape, m.wcs, tshape, twcs)
                shape, wcs = enmap.overlap(mask.shape, mask.wcs, shape, wcs)
                flat_template = enmap.zeros((3, shape[0], shape[1]), wcs)
                mask = enmap.insert(flat_template.copy()[0], mask)
                m = enmap.insert(flat_template.copy(), m)

            field_spin0 = nmt.NmtField(mask, m[:1], wcs=wcs)
            field_spin2 = nmt.NmtField(mask, m[1:], wcs=wcs,
                                       purify_b=meta.pure_B)

            fields[map_set, id_bundle] = {
                "spin0": field_spin0,
                "spin2": field_spin2
            }

        for map_name1, map_name2 in meta.get_ps_names_list(type="all",
                                                           coadd=False):
            map_set1, id_split1 = map_name1.split("__")
            map_set2, id_split2 = map_name2.split("__")
            
            ofile = f"{cells_dir}/decoupled_pcls_{map_name1}_x_{map_name2}_{id_sim:04d}.npz"
            if os.path.exists(ofile):
                continue
            
            if verbose:
                print(f"# {id_sim+1} | ({map_set1}, split {id_split1}) x "
                      f"({map_set2}, split {id_split2})")
            
            pcls = pu.get_coupled_pseudo_cls(
                    fields[map_set1, id_split1],
                    fields[map_set2, id_split2],
                    nmt_bins
                    )

            decoupled_pcls = pu.decouple_pseudo_cls(
                    pcls, inv_couplings_beamed[map_set1, map_set2]
                    )

            np.savez(ofile,
                     **decoupled_pcls, lb=nmt_bins.get_effective_ells())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no-plots", action="store_false", help="Do not make plots.") # noqa
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    parser.add_argument("--car-template", type=str, default=None, help="Path to a CAR template map (overrides meta.car_template).")
    args = parser.parse_args()
    main(args)
