# Installation

- Install SOOPERCOOL
```
git clone git@github.com:simonsobs/SOOPERCOOL.git
cd SOOPERCOOL
pip install -e .
```
`-e` option will install local src with [development/editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)


# Dive-in the config file structure
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
The first step is to create a mask from data provided through the configuration file.
