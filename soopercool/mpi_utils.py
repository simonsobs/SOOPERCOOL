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


def print_rnk0(text, rank, if_rank=0, logger=None):
    if rank == if_rank:
        if logger:
            logger.info(text)
        else:
            print(text)


def init(switch=False, logger=None):
    ''' initialize MPI set-up '''
    global _initialized, _switch
    global rank, size, comm

    exit_code = 0

    if not _initialized:
        _initialized = True
    else:
        print_rnk0("MPI is already intialized", rank, logger=logger)
        return exit_code

    if not switch:
        print_rnk0("WARNING: MPI is turned off by default. "
                   "Use mpi.init(switch=True) to initialize MPI."
                   "MPI is turned off", rank, logger=logger)
        return exit_code
    else:
        _switch = True

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    except ImportError as exc:
        sys.stderr.write("IMPORT ERROR: " + __file__ + " (" + str(exc) + "). "
                         "Could not load mpi4py. MPI will not be used.\n")
    return rank, size, comm


def is_initialized():
    global _initialized
    return _initialized


def is_mpion():
    global _switch
    return _switch


def taskrange(imax, imin=0, shift=0, logger=None):
    """
    """
    global rank, size

    if (not isinstance(imin, int) or not isinstance(imax, int)
            or not isinstance(shift, int)):
        raise TypeError("imin, imax and shift must be integers")
    elif not is_initialized():
        print_rnk0("MPI is not yet properly initialized. "
                   "Are you sure this is what you want to do?",
                   rank, logger=logger)

    if not is_mpion():
        return np.arange(imin, imax + 1)

    ntask = math.ceil((imax - imin + 1)/size)*size

    subrange = None
    if ntask <= 0:
        print_rnk0("number of task can't be zero", rank, logger=logger)
        subrange = np.arange(0, 0)  # return zero range
    else:
        if ntask != imax - imin + 1:
            print_rnk0(f"WARNING: setting ntask={ntask}", rank, logger=logger)
        perrank = ntask // size
        print_rnk0(f"Running {ntask} simulations on {size} nodes",
                   rank, logger=logger)
        subrange = np.arange(rank*perrank, (rank + 1)*perrank)

    return subrange


def distribute_tasks(size, rank, ntasks, id_start=0, logger=None):
    """
    Distributes [ntasks] tasks among [size] workers, and outputs
    the list of tasks assigned to a given rank.

    Parameters
    ----------
        size: int
            The number of workers available.
        rank: int
            The number (ID) of the current worker.
        ntasks: int
            The number of tasks.
        logger: logging.Logger
            Logging instance to write output. If None, ignore.
    Returns
    -------
        local_task_ids: list
            List with indices corresponding to the tasks assigned
            to worker [rank]
    """
    if size > ntasks:
        local_start = rank
        local_stop = rank + 1
    else:
        local_start = rank * (ntasks // size)
        local_stop = local_start + (ntasks // size)

    local_task_ids = list(range(id_start, id_start+ntasks))[local_start:local_stop]  # noqa

    if rank >= ntasks:
        local_task_ids = []

    # If there are more tasks than size and ntasks is not divisible by size,
    # there will be a set of ntasks_left < size leftover tasks. We distribute
    # one of them each to the first ntasks_left workers.
    if ntasks % size != 0 and size < ntasks:
        leftover = np.arange(ntasks)[-(ntasks % size):]
        if rank < len(leftover):
            local_task_ids.append(leftover[rank])

    print_rnk0(f"Rank {rank} has {len(local_task_ids)} tasks: "
               f"{np.array(local_task_ids, dtype=int)}",
               rank, if_rank=rank, logger=logger)
    print_rnk0(f"Total number of tasks is {ntasks}", rank, logger=logger)

    return local_task_ids
