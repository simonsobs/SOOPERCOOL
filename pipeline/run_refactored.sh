#!/usr/bin/env bash

paramfile='../paramfiles/paramfile_refactored_noisy.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Pre-processing
# Hits: here I just copied the SAT hits map to the data directory
python misc/get_binning.py --globals ${paramfile} --deltal 10
python misc/get_sat_beams.py --globals ${paramfile}
python get_analysis_mask.py --globals ${paramfile}

# Mode coupling
srun -n 10 -c 24 python simulations/generate_tf_estimation_sims.py --globals ${paramfile} --verbose --no-plots
python get_mode_coupling.py --globals ${paramfile}
srun -n 10 -c 24 python transfer/compute_pseudo_cells_tf_estimation.py --globals ${paramfile}
python transfer/compute_transfer_function.py --globals ${paramfile}
python get_full_couplings.py --globals ${paramfile}

# Data synthesis & covariance
srun -n 10 -c 24 python simulations/generate_sat_noise.py --globals ${paramfile} --verbose
python simulations/generate_mock_cmb_sky.py --globals ${paramfile} --verbose
srun -n 10 -c 24 python generate_simulations.py --globals ${paramfile} --verbose
srun -n 10 -c 24 python compute_sims_pseudo_cells.py --globals ${paramfile} --verbose
srun -n 10 -c 24 python coadd_sims_pseudo_cells.py --globals ${paramfile} --verbose
python compute_covariance_from_sims.py --globals ${paramfile}
srun -n 10 -c 24 python create_sacc_file.py --globals ${paramfile} --sims

# Data: for testing, simply copy one simulation to the data directory
# and rename it according to the parameter file
python compute_pseudo_cells.py --globals ${paramfile} --verbose
python coadd_pseudo_cells.py --globals ${paramfile}
python create_sacc_file.py --globals ${paramfile} --data

# Plot results
python sacc_plotter.py --globals ${paramfile} --sims
