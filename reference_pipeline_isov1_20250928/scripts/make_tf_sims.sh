#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=56
#SBATCH --time=00:10:00
#SBATCH --job-name=tf-sims
#SBATCH --mail-user=kevin.wolz@physics.ox.ac.uk
#SBATCH -o log-tf-sims

pix_type="car"
smooth_fwhm=30
n_sims=2
out_dir=/cephfs/soukdata/user_scratch/kwolz/end_to_end_iso/tf_sims
run_dir=/shared_home/kwolz/bbdev/SOOPERCOOL/reference_pipeline_isov1_20250928
mkdir -p $out_dir
cd $run_dir

bb_awg_scripts_dir=/shared_home/kwolz/bbdev/bb-awg-scripts


for temp in data/band_car_fejer1_1arcmin.fits data/band_car_fejer1_5arcmin.fits; do
    srun -n 2 -c 56 --cpu_bind=cores \
        python -u ${bb_awg_scripts_dir}/pipeline/misc/get_tf_simulations.py \
            --pix_type=${pix_type} \
            --smooth_fwhm=${smooth_fwhm} \
            --n_sims=${n_sims} \
            --out_dir=${out_dir} \
            --car_template_map=${temp}
done
