# Transfer functions
This is one of the key element of the SOOPERCOOL pipeline. Transfer functions depend on the type of filtering through `filtering_tag` and `kspace_tag` as described above. Once you defined these tags and pointed to the associated pure T/E/B filtered simulations, you can run the following to compute the power spectra required for TF estimation. This can be done in parallel to speed it up. The instruction below was used to run 20 pure T/E/B simulations on 1 tiger node.
```bash
srun -n 10 -c 10 --cpu_bind=cores python pipeline/transfer/compute_pseudo_cells_tf_estimation.py --globals config_file.yaml
```
Transfer function simulations can be reused. If you need to generate them yourself, you can run
```bash
python pipeline/simulations/generate_tf_estimation_sims.py --globals config_file.yaml
```
which will generate the Gaussian power-law simulations used for TF estimation according to the following settings in the yaml:
```yaml
transfer_settings:
    ...
    sim_id_start: 0
    ## Number of sims for tf estimation
    tf_est_num_sims: 20
    ## Number of sims for tf validation
    tf_val_num_sims: 20
    ## Parameters of the PL sims used for TF estimation
    power_law_pars_tf_est:
        amp: 1.0
        delta_ell: 10.
        power_law_index: 2.
    ## Optional beams applied on TF estimation sims
    # Here, paste the list of map_sets whose corresponding beams should be
    # simulated in the transfer function estimation simulations. If this is an
    # empty list, sims are convolved with a 30-arcmin Gaussian beam.
    tf_est_beams_list: []
    ## Optional beams applied on TF validation sims
    tf_val_beams_list: []
```
___
## IF RUNNING WITH FOURIER SPACE FILTERING
If you want to apply a Fourier-space filter on the maps, you'll also need to compute the associated transfer function. In this case, you'll first need to apply this $k$-space filter on simulations. This can be done running
```bash
srun -n 10 -c 10 --cpu_bind=cores python pipeline/kspace/filter_sims_kspace.py --globals config_file.yaml
```
For each seed, this will load simulations in `{filtered_map_dir}/{filtered_map_template}` as defined in the configuration file, and apply them a Fourier-space filter. If this is the only filter applied (i.e. if all filtering tags are set to `null`), then the easiest solution is to write under the `transfer_settings` section
```yaml
transfer_settings:
   ...  
  unfiltered_map_dir:
    null: /path/to/unfiltered/maps
  unfiltered_map_template:
    null: unfiltered_map_{pure_type}(...){id_sim:04d}
  filtered_map_dir:
    null: /path/to/unfiltered/maps
  filtered_map_template:
    null: unfiltered_map_{pure_type}(...){id_sim:04d}
```
This is a small workaround to define "no-filtering" with unity transfer function before applying any map-space filters.
___
Once you've computed the pure T/E/B spectra, you can estimate the power suppression induced by your set of filters (i.e. the transfer function)
```bash
python pipeline/transfer/compute_transfer_function.py --globals config_file.yaml
```
___
## K-SPACE VALIDATION
If you wish to run a quick validation using $k$-space filtering only, run
```bash
python pipeline/get_mode_coupling.py --globals config_file.yaml
srun -n 20 -c 11 --cpu_bind=cores python pipeline/transfer/validate_transfer_function_kspace.py --globals config_file.yaml
```
which filters the TF estimation simulations on the fly using a kx20 filter, computes the transfer function, generates six sets of validation simulations (CMB signal-only, white noise, power-law signal-only, each with (E+B) and B-only signal), filters those, computes the TF-corrected decoupled power spectra, and compares them with the corresponding bandpower-convolved theory. Summary plots including chi2 statistics will be available under `{output_directory}/plots/cells_tf_val_kspace`.
___
## GENERAL VALIDATION
If you wish to validate the transfer function for general filtering settings, your config file must point to existing filtered and unfiltered simulations matching the existing transfer functions saved at disk. For example, the config may look like
```yaml
transfer_settings:
  ## Path to existing transfer fucntions
  transfer_directory: /path/to/existing/transfer_functions
  ...
  ## Path to the sims for TF validation
  validation:
    unfiltered_map_dir:
      SATp1_f090_south_science: /path/to/unfiltered/cmb_sims
      SATp1_f150_south_science: /path/to/unfiltered/cmb_sims
    unfiltered_map_template:
      SATp1_f090_south_science: "cmb_4arcmin_fwhm30_sim{id_sim:04d}_CAR.fits"
      SATp1_f150_south_science: "cmb_4arcmin_fwhm30_sim{id_sim:04d}_CAR.fits"
    filtered_map_dir:
      SATp1_f090_south_science: /path/to/filtered/cmb_sims
      SATp1_f150_south_science: /path/to/filtered/cmb_sims
    filtered_map_template:
      SATp1_f090_south_science: "cmb_4arcmin_fwhm30_sim{id_sim:04d}_CAR.fits"
      SATp1_f150_south_science: "cmb_4arcmin_fwhm30_sim{id_sim:04d}_CAR.fits"
```
To compute power spectra and validate them, you then run
```bash
srun -n 20 -c 11 --cpu_bind=cores python pipeline/transfer/compute_pseudo_cells_tf_validation.py --globals config.yaml
python pipeline/transfer/validate_transfer_function.py --globals config.yaml
```
which saves validation plots under `{output_directory}/plots/cells_tf_val`.
