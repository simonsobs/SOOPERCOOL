# Installation using micromamba

These instructions show how to install `SOOPERCOOL` and `TOAST` at NERSC using
micromamba. Micromamba is a programming environment similar to conda, but 
based on C and therefore much quicker at resolving and installing packages.

### Micromamba

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
You will see some installation messages showing up and are asked to confirm 
the location of your `micromamba` environment files. Then apply the following 
settings:
```
source ~/.bashrc
micromamba config append channels conda-forge
micromamba config set channel_priority strict
```



### SOOPERCOOL environment
- Set up environment (`python=3.6` proved to work with `NaMaster`)

  ```
  micromamba create -n soopercool python=3.6 numpy scipy ipython matplotlib healpy astropy -y
  micromamba activate soopercool
  ```

- Install `GSL` from source (otherwise `NaMaster` won't recognize it)

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

- Install `NaMaster` and its dependencies

  ```
  micromamba install -c conda-forge cfitsio fftw
  micromamba install -c conda-forge namaster
  ```

- Install `soopercool` pipeline and its dependencies

  ```
  git clone git@github.com:simonsobs/so_noise_models.git
  cd so_noise_models; pip install -e .
  ```



### TOAST

- Set up environment

  ```
  micromamba create -n toast python=3.9 numpy scipy ipython matplotlib pyfftw=0.12 \
  make cmake c-compiler cxx-compiler fortran-compiler -y
  micromamba activate toast
  ```

- Install dependencies

  ```
  module load cpu PrgEnv-gnu cray-mpich craype-accel-nvidia80
  micromamba install -c conda-forge cudatoolkit
  micromamba install -c conda-forge cuda-nvcc
  env MPICC=/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1 pip install --force --no-cache-dir \
  --no-binary=mpi4py mpi4py
  ```

  (This will install mpi4py at 
  ${HOME}/micromamba/envs/toast/lib/python3.9/site-packages/mpi4py.)

  ```
  pip install pshmem posix_ipc astropy==5.2 healpy pytest
  ```

  (Don't use conda for the preceding packages; this leads to conflicts!)

- Install TOAST

  ```
  pip install --pre toast
  ```

  (`--pre` causes the latest toast3 version to get installed)

- Run toast test (passes in ~15 min):

  ```
  python -c "import toast.tests; toast.tests.run()"
  ```

  (**SOLVED:** issue regarding `posix_ipc` (`undefined symbol: shm_unlink`): Install `pshmem` and `posix_ipc` with pip, not micromamba.

- Install `spt3g` and `so3g`

  - ```
    cd ${HOME}/spt3g_software
    mkdir -p build
    cd build
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="gcc" \
      -DCMAKE_CXX_COMPILER="g++" \
      -DCMAKE_C_FLAGS="-O3 -g -fPIC" \
      -DCMAKE_CXX_FLAGS="-O3 -g -fPIC -std=c++11" \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      -DPython_EXECUTABLE:FILEPATH=$(which python3) \
      -DPYTHON_MODULE_DIR="${HOME}/toast/lib/python3.9/site-packages" \
      -DCMAKE_INSTALL_PREFIX="${HOME}/toast" \
      ..
    make -j 2 install
    ```

  - ```
    micromamba install boost libflac
    cd ~/git/OpenBLAS
    make
    make install PREFIX=""${HOME}/micromamba/envs/toast"
    cd ${HOME}/so3g
    mkdir -p build
    cd build
    cmake \
      -DCMAKE_PREFIX_PATH=""${HOME}/spt3g_software/build" \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      -DPYTHON_INSTALL_DEST="${HOME}/toast" \
      -DCMAKE_INSTALL_PREFIX="${HOME}/toast" \
      ..
    make -j 2 install
    ```

  - `pytest ~/so3g/test` passes (38 passed, 3 warnings).

- Install `sotodlib` following [Bai-Chang's instructions](https://gist.github.com/Bai-Chiang/d12bf9ae12851583f2a1ced8f3dae3bb). 

  ```
  cd ~
  git clone https://github.com/simonsobs/sotodlib.git
  cd sotodlib
  pip install -e .
  ```

  `python setup.py test` passes in ~3 min (78 tests).

  **ERROR (solved):**

  ```
  MPICH ERROR [Rank 0] [job id ] [Thu Sep 21 06:09:17 2023] [login17] - Abort(-1) (rank 0 in comm 0): MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked
   (Other MPI error)
  
  aborting job:
  MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked
  ```

  **Solution:** `module unload gpu; module load cpu cudatoolkit`

- Test script using `${HOME}/sotodlib/workflows/toast_so_sim.py`