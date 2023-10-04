# BBmaster

BBmaster is a B-modes analysis pipeline designed for Simons Observatory

## Installation

First clone the repository with

```bash
git clone git@github.com:simonsobs/BBMASTER.git
```

Then install it (via pip)

```bash
cd BBMASTER
pip install numpy # Needed to install pymaster via pip
pip install -e .
```

## Run the pipeline using `mbatch`

First prepare the data
```bash
cd data
python gen_PL_sims.py
```

First clone and install mbatch

```bash
git clone git@github.com:simonsobs/mbatch.git
cd mbatch
pip install -e .
```

and run the whole pipeline with

```bash
mbatch output test_mbatch.yml
```