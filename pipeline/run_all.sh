paramfile='../paramfiles/paramfile_SAT_256_clean.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

echo "------------------------------------------------------------"
echo "|           PREPARING METADATA AND SIMULATIONS             |"
echo "------------------------------------------------------------"

echo "Pre-processing data..."
echo "-------------------"
python pre_processer.py --globals ${paramfile} --sims --verbose
python mask_handler.py --globals ${paramfile} --plots --verbose

echo "Running mock stage for data..."
echo "------------------------------"
python mocker.py --globals ${paramfile} --verbose

echo "Running mock stage for sims..."
echo "------------------------------"
python mocker.py --globals ${paramfile} --sims --verbose

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

echo "Running cl estimation for tf estimation"
echo "---------------------------------------"
python pcler.py --globals ${paramfile} --tf_est --verbose

echo "Running transfer estimation"
echo "---------------------------"
python transfer.py --globals ${paramfile}

echo "Running cl estimation for validation"
echo "------------------------------------"
python pcler.py --globals ${paramfile} --tf_val --verbose

echo "Transfer validation"
echo "---------------------"
python transfer_validator.py --globals ${paramfile}


echo "Running pcler on data"
echo "---------------------"
python pcler.py --globals ${paramfile} --data 

echo "Running pcler on sims"
echo "---------------------"
python pcler.py --globals ${paramfile} --sims --plots


echo "Running coadder on data"
echo "---------------------"
python coadder.py --globals ${paramfile} --data 

echo "Running coadder on sims"
echo "---------------------"
python coadder.py --globals ${paramfile} --sims

echo "Running covariance estimation"
echo "-----------------------------"
python covfefe.py --globals ${paramfile}

echo "Create sacc files for sims and data"
echo "-----------------------------------"
python saccer.py --globals ${paramfile} --data
python saccer.py --globals ${paramfile} --sims

echo "Plot sacc files content"
echo "-----------------------"
python sacc_plotter.py --globals ${paramfile} --data
python sacc_plotter.py --globals ${paramfile} --sims