#!/bin/bash

if [[ $# -lt 2 ]]
then
  echo "usage: $0 <nranks> <command line>"
  exit 1
fi

NRANKS=$1

shift

source /opt/intel/bin/compilervars.sh intel64

COMMAND="mpirun "

for i in `seq 1 $NRANKS`; do
  if [[ i -ne 1 ]]; then
    COMMAND+=" : "
  fi

  HOLD='-hold'
  COMMAND+="xterm $HOLD -T $i -e \"$@ | tee $i.out\""
done

eval $COMMAND
