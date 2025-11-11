# This script illustrates how the get a filtering transfer function from
# filtered pure-T/E/B simulations as done in the SAT ISO project.

# General configurations. This example runs with SATp1 ISO-v1 preprocessing
# in the southern patch with an extra option "with_bs", meaning we flag
# bad subscans in the TODs and exclude those before mapmaking.
instruments=("satp1")
extra_configs=("")
patches=("south")


########################
## Software environment
########################

# We won't be using open MP in the following scripts, only MPI where necessary.
export OMP_NUM_THREADS=1

module use /cephfs/soukdata/software/modulefiles/
module load soconda/20241017_3.10

# Code references. The software packages needed to run this script are:
# * sotodlib (https://github.com/simonsobs/sotodlib/tree/master/sotodlib)
# * bb-awg-scripts (https://github.com/simonsobs/bb-awg-scripts/tree/main)
# * soopercool (https://github.com/simonsobs/SOOPERCOOL/tree/reference_pipeline_isov1_20250928)

# Install the above packages (on soconda, everything should be readily
# installable via `git clone [url]; cd [package];  pip install . --user`.

# Next, insert your own installation paths below. 
basedir=/shared_home/kwolz/bbdev/SOOPERCOOL/reference_pipeline_isov1_20250928
soopercool_dir=/shared_home/kwolz/bbdev/SOOPERCOOL


########################
## SOOPERCOOL yaml file
########################

# The configuration file for SOOPERCOOL is
# ./configs/soopercool_iso/paramfile_satp1_south_science_with_bs.yaml
# Open it and have a look at the documentation inside!


######################
## Running the stages
######################

# For ISO, several null splits were run to doing null tests, see the commented
# options for information. For ISO v2 (ongoing), only "science" is available
# - meaning that all data were considered. In principle, null split TFs are
# computed in exactly the same way but with other data.

# To run the code, leave this as is.
splits=(
  "science"
  # "det_in_out"      
  # "det_left_right"
  # "det_upper_lower"
  # "scan_left_right"
  # "high_low_pwv"
  # "high_low_dpwv"
  # "high_low_sun_distance"
  # "high_low_ambient_temperature"
  # "high_low_ctime"
  # "rising_setting_azimuth"
)

cd $basedir

# This runs each step for each instrument, patch, split, and extra config.
for sat in "${instruments[@]}"; do
  for patch in "${patches[@]}"; do
	for split in "${splits[@]}"; do
	  for xcfg in "${extra_configs[@]}"; do
		echo "=== Running for sat: $sat | patch: $patch | split: $split | $xcfg ==="

		paramfile="${basedir}/paramfiles/paramfile_${sat}_${patch}_${split}${xcfg}.yaml"
		echo "Using paramfile: $paramfile"

		#######################################
        ## Uncomment the steps you want to run
		#######################################

		# # Mask generation
		# python -u ${soopercool_dir}/pipeline/get_analysis_mask.py --globals ${paramfile}

		# # Bandpower binning (choose Delta_ell = 15 as in SAT ISO)
		# python -u ${soopercool_dir}/pipeline/misc/get_binning.py --globals ${paramfile} --deltal 15

		# # The TF spectrum estimation can be run in parallel.
		# # Must set ntasks = (# sims) * (# map sets) * (# map sets + 1) / 2
		# # maximum allowed cmax = (# nodes) * 112 // ntasks  (at tiger3).
		# # E.g., #nodes=1, #sims=2 and #(map sets)=1, so ntasks=2, cmax=56
		# srun -n 2 -c 56 --cpu-bind=cores python -u \
		# ${soopercool_dir}/pipeline/transfer/compute_pseudo_cells_tf_estimation.py --globals ${paramfile} --verbose

		# Filtering transfer function computation
		python -u ${soopercool_dir}/pipeline/transfer/compute_transfer_function.py --globals ${paramfile}

		# Masking mode coupling matrix
		python -u ${soopercool_dir}/pipeline/get_mode_coupling.py --globals ${paramfile}

		# Full (filtering + masking) bandpower coupling matrix
		python -u ${soopercool_dir}/pipeline/get_full_couplings.py --globals ${paramfile}

		echo "=== Done with sat: $sat | patch: $patch | split: $split | $xcfg ==="; echo
	  done
	done
  done
done

#########
## Plots
#########
# Data products can be found under {output_directory} as set in the yaml file.
# Several of them have been plotted automatically. Have a look at the
# {output_directory}/plots subdirectory to explore them:

# {output_directory}/plots
# ├── cells
# ├── cells_tf_est       (exists if TF was computed from scratch)
# ├── couplings
# ├── maps
# ├── masks
# ├── noise             
# └── transfer_functions (exists if TF was computed from scratch)