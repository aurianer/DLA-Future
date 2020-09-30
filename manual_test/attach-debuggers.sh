#!/bin/bash

if [[ $# -eq 0 ]]
then
  echo "usage: $0 <process name>"
  echo "it attaches a lldb instance to all processes with given name (using pgrep)"
  exit 1
fi

#spack load mpich

function join_by { local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}"; }

COMMANDS=()
for PID in `pgrep $@`; do
  COMMANDS+=("xterm -e lldb -s ~/lldb.script -p $PID")
done

mpirun $(join_by " : " "${COMMANDS[@]}")
