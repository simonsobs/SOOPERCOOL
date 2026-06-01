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

# Running instructions

Below we give detailed instructions on how to run the SOOPERCOOL pipeline sequentially from maps to power spectra and covariances compiled in a `SACC` file.

## Mask
The first step is to create a mask from data provided through the configuration file.
