#!/usr/bin/env bash

paramfile='../paramfiles/paramfile_wmap.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Run serially
# echo "Pre-processing real data"
# echo "-----------------------------"
# python pre_processer.py --globals ${paramfile} --verbose
# python mask_handler.py --globals ${paramfile} --plots --verbose
# python pre_processer_ext.py --globals ${paramfile} --wmap --data
# python pre_processer_ext.py --globals ${paramfile} --wmap --noise

# echo "Running filterer for data"
# echo "-------------------------"
# python filterer.py --globals ${paramfile} --data

# echo "Running mcm..."
# echo "--------------"
# python mcmer.py --globals ${paramfile} --plot


# run in parallel with salloc -N 1 -C cpu -q interactive -t 00:30:00
echo "Generating transfer sims"  # 1m20 for 30 sims
echo "-----------------------------"
srun -n 30 -c 8 python pre_processer.py --globals ${paramfile} --verbose --sims

echo "Running filterer for transfer"  # 1m20 for 30 sims
echo "-----------------------------"
srun -n 30 -c 8 --cpu_bind=cores python filterer.py --globals ${paramfile} --transfer

echo "Generating sims"
echo "-------------------------"
srun -n 25 -c 10 --cpu_bind=cores python mocker.py --globals ${paramfile} --sims

echo "Running filterer for sims"  # 2m50 for 100 sims / 10 splits
echo "-------------------------"
srun -n 25 -c 10 --cpu_bind=cores python filterer.py --globals ${paramfile} --sims

echo "Running cl estimation for tf estimation"  # 1m50 for 30 sims
echo "---------------------------------------"
srun -n 30 -c 8 --cpu_bind=cores python pcler.py --globals ${paramfile} --tf_est --verbose

echo "Running transfer estimation"
echo "---------------------------"
python transfer.py --globals ${paramfile}

echo "Running cl estimation for validation"
echo "------------------------------------"
srun -n 30 -c 8 --cpu_bind=cores python pcler.py --globals ${paramfile} --tf_val --verbose

echo "Transfer validation"
echo "---------------------"
python transfer_validator.py --globals ${paramfile}


# run in parallel with salloc -N 1 -C cpu -q interactive -t 01:00:00
echo "Running pcler on data"
echo "---------------------"
python pcler.py --globals ${paramfile} --data 

echo "Running pcler on sims"  # 5m  for 100 sims / 21 splits
echo "---------------------"
srun -n 25 -c 10 --cpu_bind=cores python pcler.py --globals ${paramfile} --sims --verbose

echo "Running coadder on data"
echo "---------------------"
python coadder.py --globals ${paramfile} --data 

echo "Running coadder on sims"  # 0m37s  for 100 sims / 21 splits
echo "---------------------"
srun -n 25 -c 10 --cpu_bind=cores python coadder.py --globals ${paramfile} --sims

echo "Running covariance estimation"
echo "-----------------------------"
python covfefe.py --globals ${paramfile}

echo "Create sacc files for sims and data"
echo "-----------------------------------"
python saccer.py --globals ${paramfile} --data
#  2m  for 100 sims / 21 splits
srun -n 25 -c 10 --cpu_bind=cores python saccer.py --globals ${paramfile} --sims