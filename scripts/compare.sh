#!/usr/bin/env bash

set -exuo pipefail

debug=0
#scaling=("strong" "weak")
scaling_list=("strong")

####################################################################################################
# TODO: edit those variables
benchmark_names_array=("nostdexec-mimalloc" "stdexec-mimalloc")
dlafpath_array=(\
    "/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/dla-future-git.202410-develop_0.7.0-oftql6yucxh5e2jkwjdvpqxxz5yjqh2w/bin" \
    "/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/dla-future-git.202410-develop_0.7.0-zh6jn257e5jy6sy62iagoqz4474is67h/bin")
squashfs_array=(
    "/capstor/scratch/cscs/aurianer/squashfs/dlaf/dlaf-nostdexec-mimalloc-latest.squashfs" \
    "/capstor/scratch/cscs/aurianer/squashfs/dlaf/dlaf-stdexec-mimalloc-latest.squashfs")

if [[ $debug == 1 ]]; then
    configurations=(mc)
else
    #configurations=(mc gpu)
    configurations=(gpu)
fi
####################################################################################################

set +x
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
NORMAL=$(tput sgr0)
set -x

if [[ $debug == 1 ]]; then
    set +x
    idx=0
    echo ""
    echo "DEBUG"
    for dlafpath in "${dlafpath_array[@]}"; do
        echo "benchmark $((idx+1))"
        echo  $dlafpath
        echo  ${benchmark_names_array[$idx]}
        echo  ${squashfs_array[$idx]}
        idx=$((idx+1))
    done
    set -x
fi

set +u
date=$(printf "%(%d-%m-%Y_%H-%M-%S)T\n" $EPOCHSECONDS)
set -u

# Check if right directory
read -p "Do you want to run the benchmark script in this directory DLAF directory: $PWD? (y/n)" answer
case $answer in
    N|n|no|No)
        echo "Please change directory to the right one"
        exit 1
        ;;
    Y|y|yes|Yes)
        ;;
esac
if [[ $# -lt 1 ]]; then
    echo "${YELLOW}Give your benchmark a name: $0 \"<benchmark_name>\"${NORMAL}"
else
    benchmark_global_name="${1}"
fi

shift

for scaling in "${scaling_list[@]}"; do
    for configuration in "${configurations[@]}"; do
        echo "Launching jobs for configuration ${configuration}:"

        # launch jobs
        idx=0
        for bench_name in "${benchmark_names_array[@]}"; do
            ./gen_dlaf_strong-$configuration.py  --benchname ${benchmark_global_name} \
                --rundir ${SCRATCH}/benchmarks/dlaf-benchmarks/${date}-${bench_name}-${configuration} \
                --dlafpath "${dlafpath_array[$idx]}" --uenv "${squashfs_array[$idx]}"
            #./gen_dlaf_weak-$configuration.py --benchname ${benchmark_global_name} \
            #    --rundir ${SCRATCH}/benchmarks/dlaf-benchmarks/${date}-${bench_name}-${configuration} \
            #    --dlafpath "${dlafpath_array[$idx]}" --uenv "${squashfs_array[$idx]}"
            idx=$((idx+1))
        done
    done
done
