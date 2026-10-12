## Null spectra compilation
Below we provide a tutorial to compile null spectra for both data and simulations (usually sign-flips) into a common archive (a `SACC` file) and get the null summary statistics.

**WARNING: this assumes that you generated sign-flip simulations and provided the appropriate paths via the SOOPERCOOL configuration file.**

Compiling spectra in a `SACC` file is relatively quick and doesn't require any parallelization. Just run
```bash
python create_null_sacc.py --globals paramfile.yaml \
                           --overwrite # (Optional) If you need to overwrite
```
or wrap it in `srun` with a single task.

Running `create_null_sacc.py` for all null-split SOOPERCOOL configuration files will generate a `SACC` files for each null test category (left/right, in/out, ...).

We provide a script to (1) concatenate all residuals in a single `SACC` and save it to disk and (2) generate summary statistic plots for the selected configuration. You should run
```bash
python null_summary.py \
    --sacc-files sacc1.fits sacc2.fits ... \ # List of SACC files
    --out-dir /path/to/dir \ # Output directory
    --lmins 0 30 ...\ # Min multipoles to loop over
    --lmaxs 100 200 ...\ # Max multipoles to loop over
    --field-pairs EE EB BB \ # Field pairs to include in summary
    --inspect # (Optional) If you just need to inspect a file
```
or wrap it in `srun` with a single task as there is no MPI support.

As for our SOOPERpower power spectrum viewer (see `pipeline/postproc/sacc_viewer`), we provide a `streamlit` web interface. On your local machine, you can launch the app with
```bash
streamlit run SOOPERzero.py /path/to/sacc.fits
```
which will open an interactive app in your browser. If you work on a cluster (e.g. `tiger`), you can run

```
streamlit run SOOPERzero.py /path/to/sacc.fits  \
    --server.headless true  \
    --server.port 8501 \
    --server.address 0.0.0.0
```
Then on local terminal, do:
```
ssh -L 8501:localhost:8501 USER@tiger3.princeton.edu
```
On an interactive node change the last one with:
```
ssh -L 8501:tiger-your_running_node:8501 ar3186@tiger3.princeton.edu
```