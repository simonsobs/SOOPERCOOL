## Beam-related tools
This component of the SOOPERCOOL pipeline provides utility scripts to reformat beam products from planet measurements and compute beam covariances.

### Reformatting beams
As of now, planet beams are provided as `.npz` files with fields
- `Bl` for the beam window function
- `omega` solid angle for normalization
- `analytic_cov_Bl` statistical covariance
- `theta1_cov_Bl` core-wing transition uncertainty
- `lmax_cov_Bl` Bessel function scaling parameter `lmax`

for each telescope, frequency-band and wafers.

To coadd beams and covariances and save them as `.dat` files, run
```bash
python reformat_beam_products.py --beam-dir /path/to/beams \
                                 --out-dir /where/to/save \
                                 --telescopes satp1,satp3 \
                                 --bands f090,f150 \
                                 --wafers ws0,ws1,ws2,ws3,ws4,ws5,ws6 \
                                 --wafer-weights 1,...,1 # weights (will be normalized)
```
This script will produce a beam file for each telescope and frequency band - what we usually call a `map_set` in SOOPERCOOL.

### Compute beam covariances
Once you run SOOPERCOOL to get beam-corrected spectra and covariances (analytic or montecarlo), you can compute beam covariance blocks to propagate beam uncertainties. You should run
```
python compute_beam_covariances.py --globals /path/to/config.yaml
```
it will save covariance block for all spectra pairs to disk which will be then used when creating the `SACC` archive.
