echo "------------------------------------------------------------"
echo "|           PREPARING BANDPOWER COUPLING MATRIX            |"
echo "------------------------------------------------------------"

paramfile='../paramfiles/paramfile_second_mcut_kevin.yaml'
echo "Running pipeline with paramfile: ${paramfile}"

echo "Pre-processing data..."
echo "-------------------"
python pre_processer.py --globals ${paramfile} --sims

#python mask_handler.py --globals ${paramfile}

echo "Running mock stage for data..."
echo "------------------------------"
python mocker.py --globals ${paramfile} 

echo "Running mcm..."
echo "--------------"
python mcmer.py --globals ${paramfile}

echo "Running filterer for transfer"
echo "-----------------------------"
python filterer.py --globals ${paramfile} --transfer

echo "Running filterer for data"
echo "-------------------------"
python filterer.py --globals ${paramfile} --data

echo "Running cl estimation for tf estimation"
echo "---------------------------------------"
python pcler.py --globals ${paramfile} --tf_est

echo "Running transfer estimation"
echo "---------------------------"
python transfer.py --globals ${paramfile}



echo "------------------------------------------------------------"
echo "|              TESTING PIPELINE WITH 2 SPLITS              |"
echo "------------------------------------------------------------"

echo "Running pcler on data"
echo "---------------------"
python pcler.py --globals ${paramfile} --data --plots

echo "Running coadder on data"
echo "---------------------"
python coadder.py --globals ${paramfile} --data --plots