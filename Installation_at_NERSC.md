#### Installation of BB MASTER at `NERSC`

In principle, a standard python environment with a working version of `NaMaster` will do the job. In a general environment (like `NERSC` or a different computing cluster), a safe and fast option is `micromamba`:

- `micromamba` installation:

  First, install `micromamba` to your computing environment using `curl`: 

  ```
  "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
  ```

  You will see some installation messages showing up and are asked to confirm the location of your `micromamba` environment files. Then apply the following settings:

  ```
  source ~/.bashrc
  micromamba config append channels conda-forge
  micromamba config set channel_priority strict
  ```

- Install and load `bbmaster` environment (`python=3.6` proved to work with `conda`-like environments)

  ```
  micromamba create -n bbmaster python=3.6 numpy scipy ipython matplotlib healpy astropy -y
  micromamba activate bbmaster
  ```

  Then, install `GSL` from source (otherwise `NaMaster` won't recognize it):

  ```
  wget https://mirror.kumi.systems/gnu/gsl/gsl-2.7.tar.gz
  tar -zxvf gsl-2.7.tar.gz
  cd gsl-2.7
  ./configure --prefix=/global/u2/k/kwolz/software/gsl
  make && make -j4 install
  echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/u2/k/kwolz/software/lib" >> ~/.bashrc
  echo "export CFLAGS='-I/global/u2/k/kwolz/software/include'" >> ~/.bashrc
  echo "export LDFLAGS='-I/global/u2/k/kwolz/software/lib'" >> ~/.bashrc
  source ~/.bashrc
  ```

  Finally, install `NaMaster` and its dependencies:

  ```
  micromamba install -c conda-forge cfitsio fftw
  micromamba install -c conda-forge namaster
  ```

- Install `bbmaster` pipeline and its dependencies:

  ```
  git clone git@github.com:simonsobs/so_noise_models.git
  cd so_noise_models; pip install -e .
  ```

  