#!/bin/bash

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 library"
  exit
fi

benchmark_lib=$1

domain_nodes=(1 2 4 8 16 32 64 96 128 192 256)
domain_ranks_per_node=(1 2)
domain_m=(10240 20480 40960)
domain_mb=(256 512)
nruns=5


function append_cmd_to_file {
  # $1 variable to be expaneded
  # $2 file to append to
  eval "echo $1" | tee -a $2
}

case $benchmark_lib in
  dlaf)
    cmd_line='~/bin/log_run srun\
      -n ${total_ranks}\
      -c ${cpus_per_rank}\
      ~/workspace/dla-future/build/miniapp/miniapp_triangular_solver\
      --m ${m}\
      --n ${n}\
      --mb ${mb}\
      --nb ${nb}\
      --grid-rows ${grid_rows}\
      --grid-cols ${grid_cols}\
      --nruns ${nruns}\
      --hpx:use-process-mask\
      --check-result last'
    ;;
  slate)
    cmd_line='MPICH_MAX_THREAD_SAFETY=multiple\
      ~/bin/log_run srun\
      -n ${total_ranks}\
      -c ${cpus_per_rank}\
      $PROJECT/libraries/slate/build/test/slate_test trsm\
      --dim ${m}x${n}x0\
      --nb ${nb}\
      --alpha 2\
      --p ${grid_rows}\
      --q ${grid_cols}\
      --repeat ${nruns}\
      --check n\
      --ref n\
      --type d'
    ;;
  dplasma)
    # WARNING: dplasma does not support nruns
    cmd_line+='~/bin/log_run srun\
      -n ${total_ranks}\
      -c ${cpus_per_rank}\
      $PROJECT/libraries/dplasma/build/tests/testing_dtrsm\
      -M ${m}\
      -N ${n}\
      --MB ${mb}\
      --NB ${nb}\
      --grid-rows ${grid_rows}\
      --grid-cols ${grid_cols}\
      -c 36\
      -v'
    ;;
  scalapack)
    domain_ranks_per_node=(36)
    domain_mb=(128)
    cmd_line+='~/bin/log_run srun\
      -n ${total_ranks}\
      -c ${cpus_per_rank}\
      /project/csstaff/rasolca/libevs/test_scalapack_pdtrsm\
      -m ${m}\
      -n ${n}\
      -b ${mb}\
      --rep ${nruns}'
    ;;
  *)
    echo "unknown $benchmark_lib"
    exit 1
esac

for nodes in ${domain_nodes[*]}; do

	filename=run_trsolv_${benchmark_lib}-`printf '%03d' $nodes`
	(cat <<- HEREDOC
		#!/bin/bash -l

		#SBATCH --job-name=${nodes}-trsolv-$benchmark_lib
		#SBATCH --time=02:00:00
		#SBATCH --nodes=$nodes
		#SBATCH --partition=normal
		#SBATCH --constraint=mc

		module load intel

		HEREDOC
	) | tee $filename

	for ranks_per_node in ${domain_ranks_per_node[*]}; do

		total_ranks=$(($nodes * $ranks_per_node))

		case $ranks_per_node in
      1)
				cpus_per_rank=72;;
			2)
				cpus_per_rank=36;;
      36)
        cpus_per_rank=1;;
      *)
        echo "ranks_per_node=$ranks_per_node not implemented"
        exit 1
        ;;
		esac

		grid_size=(`~/scripts/factors.py --policy maximize-square-cols ${total_ranks}`)
		grid_rows=${grid_size[0]}
		grid_cols=${grid_size[1]}

		for mb in ${domain_mb[*]}; do
			nb=${mb}

			for m in ${domain_m[*]}; do
				domain_n=($(($m/2)) $m)

				for n in ${domain_n[*]}; do
          
          case $benchmark_lib in
            dplasma)
              # WARNING: dplasma does not support binding correctly
              if [[ $ranks_per_node -ne 1 ]]
              then
                continue
              fi

              # WARNING: dplasma does not support nruns
              for i in `seq $nruns`
              do
                append_cmd_to_file "$cmd_line" $filename
              done
              echo -e '\n' | tee -a $filename
              ;;
            *)
              append_cmd_to_file "$cmd_line" $filename
          esac

				done
			done
		done
	done
done
