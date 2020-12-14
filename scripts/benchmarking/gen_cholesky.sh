#!/bin/bash

domain_nodes=(1 2 4 8 16 32 64 96 128 192 256)
domain_ranks_per_node=(1 2)
domain_block_size=(256 512)
domain_matrix_size=(10240 20480 40960)

for nodes in ${domain_nodes[*]}; do

	filename=run_cholesky_`printf '%03d' $nodes`
	(cat <<- HEREDOC
		#!/bin/bash -l

		#SBATCH --job-name=${nodes}-cholesky
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
			"1")
				cpus_per_rank=72;;
			"2")
				cpus_per_rank=36;;
		esac

		grid_size=(`~/scripts/factors.py ${total_ranks}`)
		grid_rows=${grid_size[0]}
		grid_cols=${grid_size[1]}

		for block_size in ${domain_block_size[*]}; do
			for matrix_size in ${domain_matrix_size[*]}; do

				cmd_line="~/bin/log_run srun\
					-n ${total_ranks}\
					-c $cpus_per_rank\
					~/workspace/dla-future/build/miniapp/miniapp_cholesky
					--matrix-size ${matrix_size}\
					--block-size ${block_size}\
					--grid-rows ${grid_rows}\
					--grid-cols ${grid_cols}\
					--nruns 5\
					--hpx:use-process-mask\
					--check-result last"

				echo $cmd_line | tee -a $filename

			done
		done
	done
done
