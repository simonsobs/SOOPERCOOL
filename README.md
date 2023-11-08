# Installation

- [Install mbatch](https://github.com/simonsobs/mbatch#installing)

- Install BBMASTER
```
git clone git@github.com:simonsobs/BBMASTER.git
cd BBMASTER
pip install -e .
```
`-e` option will install local src with [development/editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)

# Run test
- Prepare input files
```
cd data
python get_bandpower_edges.py
python gen_PL_sims.py
gzip -dk mask_binary.fits.gz
cd ..
```

- run `test_mbatch.yml`
```
mbatch project_name test_mbatch.yml
```
