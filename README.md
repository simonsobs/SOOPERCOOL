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

## Run the pipeline sequentially

First prepare the data
```bash
cd data
python gen_PL_sims.py
cd ..
```

Then compute the mode coupling matrices
```bash
python pipeline/mcmer.py --globals data/globals.yml --output-dir results --plot
```

```bash
python pipeline/filterer_mcut.py --globals data/globals.yml --first-sim 0 --num-sims 200 --m-cut 30 --sim-sorter pl_sim_names --output_dir results
```

```bash
python pipeline/pcler.py --globals data/globals.yml --first-sim 0 --num-sims 200 --sim-sorter pl_sim_names_EandB --sim-type input --output-dir results
```

```bash
python pipeline/pcler.py --globals data/globals.yml --first-sim 0 --num-sims 200 --sim-sorter pl_sim_names_EandB --sim-type filtered --output-dir results
```

```bash
python pipeline/pcler.py --globals data/globals.yml --first-sim 0 --num-sims 200 --sim-sorter val_sim_names --sim-type input --output-dir results
```

```bash
python pipeline/pcler.py --globals data/globals.yml --first-sim 0 --num-sims 200 --sim-sorter val_sim_names --sim-type filtered --output-dir results
```

```bash
python pipeline/transfer.py --globals data/globals.yml --use-theory --output-dir results
```

```bash
python pipeline/pcler.py --globals data/globals.yml --first-sim 0 --num-sims 200 --sim-sorter val_sim_names --sim-type decoupled --correct-transfer --output-dir results
```

```bash
python pipeline/transfer_validator.py --globals data/globals.yml --transfer-threshold 0.05 --output-dir results
```

## Running using `mbatch`

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