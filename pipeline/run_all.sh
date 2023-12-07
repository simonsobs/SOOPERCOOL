paramfile='/global/homes/k/kwolz/bbdev/SOOPERCOOL/paramfiles/paramfile_mcut_kevin.yaml'

# Load the environment module
module use --append /pscratch/sd/s/susannaz/conda_envs/master_env/modulefiles
module load master_env/0.0.3

echo "Running pipeline with paramfile: ${paramfile}"

echo "------------------------------------------------------------"
echo "|           PREPARING METADATA AND SIMULATIONS             |"
echo "------------------------------------------------------------"

echo "Pre-processing data..."
echo "-------------------"
python pre_processer.py --globals ${paramfile} --sims
python mask_handler.py --globals ${paramfile}

echo "Running mock stage for data..."
echo "------------------------------"
python mocker.py --globals ${paramfile} 

echo "Running mock stage for sims..."
echo "------------------------------"
python mocker.py --globals ${paramfile} --sims

echo "Running mcm..."
echo "--------------"
python mcmer.py --globals ${paramfile} --plot


echo "------------------------------------------------------------"
echo "|             FILTERING SIMULATIONS AND DATA               |"
echo "------------------------------------------------------------"

echo "Running filterer for transfer"
echo "-----------------------------"
python filterer.py --globals ${paramfile} --transfer
# Uncomment line below (and comment above) to use MPI
# e.g. if running with toast filtering
#OMP_NUM_THREADS=2 mpirun -n 4 python filterer.py --globals ${paramfile} --transfer

echo "Running filterer for sims"
echo "-------------------------"
python filterer.py --globals ${paramfile} --sims
#OMP_NUM_THREADS=2 mpirun -n 4 python filterer.py --globals ${paramfile} --sims

echo "Running filterer for data"
echo "-------------------------"
python filterer.py --globals ${paramfile} --data
#OMP_NUM_THREADS=2 mpirun -n 4 python filterer.py --globals ${paramfile} --data


echo "------------------------------------------------------------"
echo "|                 COMPUTING POWER SPECTRA                  |"
echo "------------------------------------------------------------"

# echo "Switching to environment with NaMaster"
# micromamba deactivate; micromamba activate <namaster_env>

echo "Running cl estimation for tf estimation"
echo "---------------------------------------"
python pcler.py --globals ${paramfile} --tf_est

echo "Running transfer estimation"
echo "---------------------------"
python transfer.py --globals ${paramfile}

echo "Running cl estimation for validation"
echo "------------------------------------"
python pcler.py --globals ${paramfile} --tf_val

echo "Transfer validation"
echo "---------------------"
python transfer_validator.py --globals ${paramfile}

echo "Running pcler on data"
echo "---------------------"
python pcler.py --globals ${paramfile} --data --plots

echo "Running pcler on sims"
echo "---------------------"
python pcler.py --globals ${paramfile} --sims

echo "Running coadder on data"
echo "---------------------"
python coadder.py --globals ${paramfile} --data --plots

echo "Running coadder on sims"
echo "---------------------"
python coadder.py --globals ${paramfile} --sims

echo "Running coadder on sims"
echo "---------------------"
python coadder.py --globals ${paramfile} --sims
