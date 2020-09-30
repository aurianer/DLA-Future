#!/bin/bash

if [[ $# -ne 1 ]]
then
  echo "usage: $0 <nranks>"
  exit 1
fi

nranks=$1

PREFIX=$HOME/workspace/repos/dla-future/manual_test/

# PARSE ALL THE OUTPUTs
$PREFIX/00-parse_output.sh $nranks

# GENERATE NUMPY DATA
$PREFIX/01-generate_numpy.sh raw_output.data > generator.py
python generator.py

# CHECK THE RESULTS
$PREFIX/02-check_example_refactoring.py

# (OPTIONAL) clean up
#$PREFIX/03-clean.sh
