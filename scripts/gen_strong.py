#!/usr/bin/env python3

# This file is an example on how to use the miniapp module.
# Please do not add gen scripts used for benchmarks into the source repository,
# they should be kept with the result produced.

import argparse
import miniapps as mp
import systems

system = systems.cscs["daint-mc"]

libpaths = {
    "dlaf": "<path_to_dlaf>",
    "dplasma": "<path_to_dplasma>",
    "slate": "<path_to_slate>",
    "scalapack-libsci": "<path_to_libsci_miniapp>",
    "scalapack-mkl": "<path_to_mkl_miniapp>",
}

run_dir = f"~/ws/runs"

time = 400  # minutes
nruns = 10
nodes_arr = [1, 2, 4, 8, 16, 32]

parser = argparse.ArgumentParser(description="Run strong scaling benchmarks.")
parser.add_argument(
    "--debug",
    help="Don't submit jobs, only create job scripts instead.",
    action="store_true",
)
parser.add_argument(
    "--libs",
    help="Run miniapps for these libraries.",
    nargs="+",
    default=["scalapack-mkl", "scalapack-libsci", "dlaf", "slate", "dplasma"],
    choices=list(libpaths.keys()),
)
args = parser.parse_args()

debug = args.debug

run_mkl = "scalapack-mkl" in args.libs
run_libsci = "scalapack-libsci" in args.libs
run_dlaf = "dlaf" in args.libs
run_slate = "slate" in args.libs
run_dp = "dplasma" in args.libs

# Example #1: Cholesky strong scaling with DLAF:
# Note params entries can be a list or a single value (which is automatically converted to a list of a value).
# The following benchmark is executed for these cases (using (m_sz, mb_sz) notation):
# (10240, 256)
# (10240, 512)
# (20480, 256)
# (20480, 512)
# for rpn = 1 and 2
# (5120, 64)
# (5120, 128)
# only for rpn = 2

path_dlaf = "/apps/daint/UES/aurianer/build/dlafuture/build_dlaf_hpx_transform_mpi"

if run_dlaf:
    run = mp.StrongScaling(system, "Cholesky_strong", nodes_arr, time)
    run.add(
        mp.chol,
        "dlaf",
        path_dlaf,
        {"rpn": [1, 2], "m_sz": [10240, 20480], "mb_sz": [256, 512]},
        nruns,
    )
#    run.add(
#        mp.chol,
#        "dlaf",
#        path_dlaf,
#        {"rpn": 2, "m_sz": 5120, "mb_sz": [64, 128]},
#        nruns,
#    )
    run.submit(run_dir, "job_chol_dlaf", debug=debug)

## Example #3: Trsm strong scaling with DLAF:
## Note: n_sz = None means that n_sz = m_sz (See miniapp.trsm documentation)
#
#if run_dp:
#    run = mp.StrongScaling(system, "Trsm_strong", nodes_arr, time)
#    run.add(
#        mp.trsm,
#        "dlaf",
#        path_dlaf,
#        {"rpn": 1, "m_sz": [10240, 20480], "mb_sz": [256, 512], "n_sz": None},
#        nruns,
#    )
#    run.submit(run_dir, "job_trsm_dlaf", debug=debug)

### Example #4: Compare two versions:
##
##if run_dlaf:
##    run = mp.StrongScaling(system, "Cholesky_strong", nodes_arr, time)
##    run.add(
##        mp.chol,
##        "dlaf",
##        path_dlaf_master,
##        {"rpn": [1, 2], "m_sz": [10240, 20480], "mb_sz": [256, 512]},
##        nruns,
##        suffix="V1",
##    )
##    run.add(
##        mp.chol,
##        path_dlaf_transform_mpi,
##        "<path_V2>",
##        {"rpn": [1, 2], "m_sz": [10240, 20480], "mb_sz": [256, 512]},
##        nruns,
##        suffix="V2",
##    )
##    run.submit(run_dir, "job_comp_dlaf", debug=debug)
