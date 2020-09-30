nranks=$1

filename=raw_output.data

cat [1-$nranks].out | egrep '^(matA|T[0-9]+|Z)[[:space:]\[]' | sort | uniq > $filename
