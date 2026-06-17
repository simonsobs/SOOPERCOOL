# Installation

- Install SOOPERCOOL
```
git clone git@github.com:simonsobs/SOOPERCOOL.git
cd SOOPERCOOL
pip install -e .
```
`-e` option will install local src with [development/editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)


# Dive-in the config file basic structure
All SOOPERCOOL scripts will interact with a YAML configuration file. The most important section defines what we call a `map_set`. This defines a collection of maps (bundles), with independent noise realizations to build our cross-bundle power spectrum estimator. 

Let's have a quick look at this block structure
```yaml
map_sets:
  map_set1: # This will be the label associated to this collection of maps
    map_dir: # Where to look for maps
    beam_dir: # Where to look for beams
    # {id_bundle} defines where the bundle index will be inserted
    # {map|hits} define the string for the data and hitmaps
    # Can be alternatively {map|weights} or any other
    # string depending on your naming covention
    map_template: fname_{id_bundle}_{map|hits}.fits 
    beam_file: fname_beam.dat # Path to the beam product (l, bl)
    n_bundles: n_bundles # Number of bundles
    freq_tag: fXXX # If we need this info to coadd frequencies
    # Any tag to identify a common experiment.
    # Any two map_sets with the same exp_tag are assumed to
    # have correlated noise and auto-bundle spectra
    # are discarded in the estimator.
    exp_tag: exp_tag
    # Any tag to identify the hit/weight map
    # to be use to weight the mask when
    # accounting for inhomogeneous noise
    # in the covariance. These are defined later on
    # in the covariance block under `hits_files`
    hits_tag: hits_tag
    filtering_tag: ftag # Tag to match map_set and filtering type
    kspace_tag:  ktag # Tag to match map_set and Fourier filtering type
    # You can also set any of the two tags above to `null`
```

Above we just illustrated the configuration file structure with one map_set, but you can have as many as you want. Let's focus now on filtering tags that are a key feature of SOOPERCOOL.

These tags are used to attach a map_set to a given filtering that has been applied to the data (`filtering_tag`) or to be applied to it (`kspace_tag`). You then define these tags in the `transfer_settings` section of the configuration file. Transfer functions will be then computed from the inputs provided in the configuration file as below.
```yaml
transfer_settings:
  tf_est_num_sims: X # Number of simulations for TF
  unfiltered_map_dir:
    # Path to unfiltered maps
    ftag1: /path/to/maps
    ftag2: /path/to/maps
  unfiltered_map_template:
    # fname template for unfiltered maps
    # should contain {pure_type} and {id_sim}
    # to know where to replace these values in
    # the file name
    ftag1: fname_{pure_type}(...){id_sim:04d}
    ftag2: fname_{pure_type}(...){id_sim:04d}
  filtered_map_dir:
    # Path to filtered maps
    ftag1: /path/to/maps
    ftag2: /path/to/maps
  filtered_map_template:
    # fname template for filtered maps
    ftag1: fname_{pure_type}(...){id_sim:04d}
    ftag2: fname_{pure_type}(...){id_sim:04d}
```
For Fourier-filter settings, you also define them (if needed) in the `transfer_settings` section. This is flexible enough to define multiple Fourier-space filters if needed.
```yaml
transfer_settings:
  kspace_pars:
    ktag1:
      dkx: int
      dky: int
      type: sharp/cosine
```

# Running instructions

Below we give detailed instructions on how to run the SOOPERCOOL pipeline sequentially from maps to power spectra and covariances compiled in a `SACC` file.

## Mask
The first step is to create a mask from data provided through the configuration file. The relevant config file section is `masks` with inputs provided as below
```yaml
masks:
  analysis_mask: /path/to/mask.fits # Which mask to use. Useful for reruns.
  galactic_mask: /path/to/mask.fits # Galactic mask (binary)
  point_source_mask: /path/to/mask.fits # PS mask (binary)
  external_mask: /path/to/mask.fits # Any other binary mask
  box_mask: null # you can provide box coordinates to restrict the binary mask
  use_weights: bool # weight binary with hits/weights
  apod_radius: 10.0 # Apodization radius (degrees)
  apod_radius_point_source: 1.0 # Apodization radius (degrees)
  apod_type: "C1" # Pick an apo_type
```
You can then create a mask running the following 
```bash
python pipeline/get_analysis_mask.fits --globals config_file.yaml
```
Be aware that this will create mask products in the SOOPERCOOL output directory, check that you are using the correct analysis mask in the configuration file before running the pipeline.

## Mode-coupling matrices
Once you generated a mask you can then pre-compute and save the NaMaster mode coupling matrices with
```bash
python pipeline/get_mode_coupling.py --globals config_file.yaml
```

## Transfer functions
This is one of the key element of the SOOPERCOOL pipeline. Transfer functions depend on the type of filtering through `filtering_tag` and `kspace_tag` as described above. Once you defined these tags and pointed to the associated pure T/E/B filtered simulations, you can run the following to compute the power spectra required for TF estimation. This can be done in parallel to speed it up. The instruction below was used to run 20 pure T/E/B simulations on 1 tiger node.
```bash
srun -n 10 -c 10 --cpu_bind=cores python pipeline/transfer/compute_pseudo_cells_tf_estimation.py --globals config_file.yaml
```
___
**IF RUNNING WITH FOURIER SPACE FILTERING**
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

## Power spectra
From this point, getting power spectra is quite straightforward, just run sequentially
```bash
python pipeline/compute_pseudo_cells.py --globals config_file.yaml
python pipeline/coadd_pseudo_cells.py --globals config_file.yaml
```
This will compute all cross-bundle pairs, and coadd them into cross, noise and auto spectra. These will be corrected for the mask mode-coupling and transfer function.

## Covariances
From this point we only need to estimate covariances. To pre-compute and save data products required for analytic covariances you can run
```bash
python pipeline/prepare_cov_inputs.py --globals config_file.yaml
srun -n 12 -c 8 --cpu_bind=cores python pipeline/precompute_cov_couplings.py --globals config_file.yaml
```
The second script will save a large amount of covariance couplings (for signal and noise cross terms) and therefore we advice to restrict the analysis lmax and to make sure you have enough disk space. These files can be deleted after computing covariances.

Running covariance is then straightforward
```bash
srun -n 12 -c 8 --cpu_bind=cores python pipeline/compute_covariance.py --globals config_file.yaml
```
and will save all covariance blocks.

Instructions for montecarlo covariances will be updated soon as these scripts need to be refurbished.

## How to create a SACC file
Compile all spectra and covariances you computed in a `SACC` file used as an input to the likelihood by running
```bash
srun -n 12 -c 8 --cpu-bind=cores python pipeline/create_sacc_file_analytic.py --globals config_file.yaml --data
```
A similar recipe to do it with either MC/analytic covariance will be described soon, these scripts also need a small refactoring.
