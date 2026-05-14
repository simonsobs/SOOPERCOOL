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
streamlit run sacc_viewer/SOOPERpower.py
```
which will open a local session in your default browser (don't forget to use tunelling to view it if you're on a cluster). You can then select the `sacc` file you want to visualize through an (ugly) GUI interface. This will open a web interface to visualize spectra for different `map_set` pairs as well as an additional tab to show power spectra residuals, computing $\chi^2$ and PTEs on the fly.

Otherwise you can run it on a cluster (eg tiger3) without relying on the file dialog window with (pass as many sacc files as you wish and you can label them with the `:label` syntax):
```
streamlit run sacc_viewer/SOOPERpower.py cl_and_cov_sacc1.fits:label_for_sacc1 cl_and_cov_sacc2.fits:label_for_sacc2  --server.headless true  --server.port 8501  --server.address 0.0.0.0
```
Then on local terminal, do:
```
ssh -L 8501:localhost:8501 USER@tiger3.princeton.edu
```
On an interactive node change the last one with:
```
ssh -L 8501:tiger-your_running_node:8501 ar3186@tiger3.princeton.edu
```

### EB angles fit and rotation of spectra
We provide a post-processing python script to fit for EB mixing angles for each of the `map_sets` defined in the `sacc` file provided. It then write down a new `sacc` file where polarization has been rotated back using the best fit angles.

Instructions to run it are
```
python fit_EB_angles.py --sacc-file /path/to/sacc \
                        --lmin-fit ... [int] \
                        --lmax-fit ... [int] \
                        --lmax-sacc ... [int]
                        --map-sets-to-fit map_set1,map_set2,...
```
The parameter `lmax-sacc` is used to cut sacc files before building the full rotation matrix to rotate it and save it back. It is usually useful to set it to a relatively low value (e.g. 650) to make this step less computationally expensive.

### EE relative amplitude correction
We provide a post-processing python script to fit for calibration amplitudes from the EE power spectra. You need to select a list of `map_sets` among the `sacc` tracers to be the reference of this calibration procedure and list the `map_sets` you would like to get the calibration amplitude of. This script provides support for jointly fitting calibration and beam fwhms via the optional flag `--no-beam-fit`

Instructions to run it are
```
python fit_cals_and_beams.py --sacc_file /path/to/sacc \
                             --lmin-fit ... [int] \
                             --lmax-fit ... [int] \
                             --lmax-sacc ... [int] \
                             --map-sets-refs map_set_ref1,map_set_ref2,... \
                             --map-sets-to-fit map_set1,map_set2,... \
                             --no-beam-fit [optional]
```
where `map-sets-refs` and `map-sets-to-fit` should have the same length to be able to select a different reference map for each `map_set` we want to calibrate.