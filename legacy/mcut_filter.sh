paramfile="../paramfiles/paramfile_refactored.yaml"
map_dir="outputs_refactor/tf_est_sims"
map_template="pureB_power_law_tf_est_{sim_id:04d}_SATp3_f150.fits"
map_template="cmb_maps_sat1_f090_bundle0_{sim_id:04d}.fits"
sim_ids="0,99"
out_dir="mcut30_filtered_tf_est_sims"
filter_tag="mcut30"

python filtering/filter_TQU_map.py --globals ${paramfile} --map-dir ${map_dir} \
                                   --map-template ${map_template} --sim-ids ${sim_ids} \
                                   --out-dir ${out_dir} --filter-tag ${filter_tag}