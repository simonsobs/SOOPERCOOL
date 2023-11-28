echo "------------------------------------------------------------"
echo "|           PREPARING BANDPOWER COUPLING MATRIX            |"
echo "------------------------------------------------------------"

paramfile='../paramfiles/paramfile_mcut_kevin.yaml'
echo "Running pipeline with paramfile: ${paramfile}"

echo "Pre-processing data..."
echo "-------------------"
python pre_processer.py --globals ${paramfile} --sims

#python mask_handler.py --globals ${paramfile}

echo "Running mock stage for data..."
echo "------------------------------"
python mocker.py --globals ${paramfile} 
#echo "Running mock stage for sims..."
#echo "------------------------------"
#python mocker.py --globals ${paramfile} --sims

echo "Running mcm..."
echo "--------------"
python mcmer.py --globals ${paramfile}

#  echo "Switching to environment 'toast'"
# micromamba deactivate; micromamba activate toast

echo "Running filterer for transfer"
echo "-----------------------------"
python filterer.py --globals ${paramfile} --transfer
echo "Running filterer for sims"
echo "-------------------------"
#python filterer.py --globals ${paramfile} --sims
echo "Running filterer for data"
echo "-------------------------"
python filterer.py --globals ${paramfile} --data

echo "Running cl estimation for tf estimation"
# micromamba deactivate; micromamba activate bbpower
# echo "Switching to environment 'bbpower'."
echo "---------------------------------------"
python pcler.py --globals ${paramfile} --tf_est

echo "Running transfer estimation"
echo "---------------------------"
python transfer.py --globals ${paramfile}

# echo "Running cl estimation for validation"
# echo "------------------------------------"
# python pcler.py --globals ${paramfile} --tf_val

# echo "Transfer validation"
# echo "---------------------"
# python transfer_validator.py --globals ${paramfile}


echo "------------------------------------------------------------"
echo "|              TESTING PIPELINE WITH 2 SPLITS              |"
echo "------------------------------------------------------------"

echo "Running pcler on data"
echo "---------------------"
python pcler.py --globals ${paramfile} --data --plots
#echo "Running pcler on sims"
#echo "---------------------"
#python pcler.py --globals ${paramfile} --sims

echo "Running coadder on data"
echo "---------------------"
python coadder.py --globals ${paramfile} --data --plots
# echo "Running coadder on sims"
# echo "---------------------"
# python coadder.py --globals ${paramfile} --sims