paramfile='../paramfiles/paramfile_planck.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

echo "------------------------------------------------------------"
echo "|           PREPARING METADATA AND SIMULATIONS             |"
echo "------------------------------------------------------------"

echo "Pre-processing data..."
echo "-------------------"
python mask_handler.py --globals ${paramfile}
python pre_processer_planck.py --globals ${paramfile} --plots --sims # --noise
