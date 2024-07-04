#!/usr/bin/env python
import sys
import numpy as np
import math

# set the default value
_initialized = False
_switch = False
rank = 0
size = 1
comm = None


def print_rnk0(text, rank):
    if rank == 0:
        print(text)


def init(switch=False):
    ''' initialize MPI set-up '''
    global _initialized, _switch
    global rank, size, comm

    exit_code = 0

    if not _initialized:
        _initialized = True
    else:
        print("MPI is already intialized")
        return exit_code

    if not switch:
        print("WARNING: MPI is turned off by default. "
              "Use mpi.init(switch=True) to initialize MPI")
        print("MPI is turned off")
        return exit_code
    else:
        _switch = True

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print("MPI: rank %d is initalized" % rank)

    except ImportError as exc:
        sys.stderr.write("IMPORT ERROR: " + __file__ + " (" + str(exc) + "). "
                         "Could not load mpi4py. MPI will not be used.\n")


def is_initialized():
    global _initialized
    return _initialized


def is_mpion():
    global _switch
    return _switch


def taskrange(imax, imin=0, shift=0):
    """
    """
    global rank, size

    if (not isinstance(imin, int) or not isinstance(imax, int)
            or not isinstance(shift, int)):
        raise TypeError("imin, imax and shift must be integers")
    elif not is_initialized():
        print("MPI is not yet properly initialized. "
              "Are you sure this is what you want to do?")

    if not is_mpion():
        return np.arange(imin, imax + 1)

    ntask = math.ceil((imax - imin + 1)/size)*size

    subrange = None
    if ntask <= 0:
        print_rnk0("number of task can't be zero", rank)
        subrange = np.arange(0, 0)  # return zero range
    else:
        if ntask != imax - imin + 1:
            print_rnk0(f"WARNING: setting ntask={ntask}", rank)
        perrank = ntask // size
        print_rnk0(f"Running {ntask} simulations on {size} nodes", rank)
        subrange = np.arange(rank*perrank, (rank + 1)*perrank)

    return subrange
