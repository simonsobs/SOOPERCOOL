## Post-processing of power spectra

The end product of the power spectrum pipeline (SOOPERCOOL) is a `sacc` file, an archive of all spectra and associated covariance blocks. This module does all the bookkeeping internally and are easy to manipulate.

This `postproc` directory contains utility scripts to visualize and apply post-processing operations to a selected `sacc` file.

### SACC visualizer
First we provide a `streamlit` app `sat_spectra_viewer.py`. This requires the `streamlit` package to be installed via
```bash
pip install streamlit
```
as well as `tkinter` which you can install on Debian/Ubuntu via
```
sudo apt-get install python3-tk
```
if not provided in your default python install. You can then run the app with
```
streamlit run sacc_viewer/sat_spectra_viewer.py
```
which will open a local session in your default browser (don't forget to use tunelling to view it if you're on a cluster). You can then select the `sacc` file you want to visualize through an (ugly) GUI interface. This will open a web interface to visualize spectra for different `map_set` pairs as well as an additional tab to show power spectra residuals, computing $\chi^2$ and PTEs on the fly.

### EB angles fit and rotation of spectra
We provide a post-processing python script to fit for EB mixing angles for each of the `map_sets` defined in the `sacc` file provided. It then write down a new `sacc` file where polarization has been rotated back using the best fit angles.

Instructions to run it are
```
python fit_EB_angles.py --sacc-file /path/to/sacc \
                        --lmin-fit ... [int] \
                        --lmax-fit ... [int] \
                        --lmax-sacc ... [int]
```
The parameter `lmax-sacc` is used to cut sacc files before building the full rotation matrix to rotate it and save it back. It is usually useful to set it to a relatively low value (e.g. 650) to make this step less computationally expensive.

### EE relative amplitude correction
We provide a post-processing python script to fit for calibration amplitudes from the EE power spectra. You need to select a `map_set` among the `sacc` tracers to be the reference of this calibration procedure and list the `map_sets` you would like to get the calibration amplitude of.

Instructions to run it are
```
python fit_EE_cals.py --sacc-file /path/to/sacc \
                      --lmin-fit ... [int] \
                      --lmax-fit ... [int] \
                      --lmax-sacc ... [int] \
                      --map-set-ref name_of_reference_map_set \
                      --map-sets-to-fit map_set1,map_set2,...
```