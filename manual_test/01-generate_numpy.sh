#!/bin/bash

inputfile=$1

TEMPLATE="np.save(\"{}\", {})"
GENERATED=`grep 'np\.*' $inputfile | egrep -o '^\S+' | xargs -I{} sh -c "printf '$TEMPLATE\n'"`

echo "import numpy as np"
cat $inputfile
printf "$GENERATED"
