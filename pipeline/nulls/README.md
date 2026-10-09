## Null spectra compilation
Below we provide a tutorial to compile null spectra for both data and simulations (usually sign-flips) into a common archive (a `SACC` file) and get the null summary statistics.

**WARNING: this assumes that you generated sign-flip simulations and provided the appropriate paths via the SOOPERCOOL configuration file.**

Compiling spectra in a `SACC` file is relatively quick and doesn't require any parallelization. Just run
```bash
python create_null_sacc.py --globals paramfile.yaml \
                           --overwrite # (Optional) If you need to overwrite
```
or wrap it in `srun` with a single task.