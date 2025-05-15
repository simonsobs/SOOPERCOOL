paramfile="../paramfiles/paramfile_refactored.yaml"
atomic_maps_dir="d"
atomics_list="outputs_refactor_high_dec/bundled_maps/atomic_map_list_f090_bundle0.txt"
context="d"

map_dir="outputs_refactor/tf_est_sims"
map_template="pureB_power_law_tf_est_{sim_id:04d}_SATp3_f150.fits"
sim_ids="0,9"

python filtering/filter_sotodlib.py \
    --globals ${paramfile} \
    --atomic-maps-dir ${atomic_maps_dir} \
    --atomics-list ${atomics_list} \
    --context ${context} \
    --map-dir ${map_dir} \
    --map-template ${map_template} \
    --sim-ids ${sim_ids} \