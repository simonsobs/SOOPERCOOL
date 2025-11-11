#!/bin/bash -l

#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=satp1-tf-sims-filtering
#SBATCH --mail-user=kevin.wolz@physics.ox.ac.uk
#SBATCH -o log-satp1-tf-sims-filtering

set -e

module use /cephfs/soukdata/software/modulefiles/
module load soconda/20241017_3.10
SOTODLIB_DIR=/shared_home/kwolz/software/sotodlib-iso-sat-review-v1
cd $SOTODLIB_DIR
#pip install . --user

run_dir=/shared_home/kwolz/bbdev/SOOPERCOOL/reference_pipeline_isov1_20250928
cd $run_dir

# Log file
tel="satp1"
log="./log_${tel}_tf_sims"

soopercool_config=paramfiles/paramfile_satp1_south_science.yaml
filtering_config=paramfiles/filtering_config_satp1.yaml
bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts

srun -n 112 -c 4 --cpu_bind=cores \
python -u ${bb_awg_scripts_dir}/pipeline/filtering/filter_sims_sotodlib.py \
    --config_file $filtering_config --sim_ids 0,1 > "${log}_filter" 2>&1

srun -n 32 -c 14 --cpu-bind=cores \
python -u ${bb_awg_scripts_dir}/pipeline/filtering/coadd_filtered_sims.py \
    --config_file $filtering_config -sim_ids 0,1 > "${log}_coadd" 2>&1filter