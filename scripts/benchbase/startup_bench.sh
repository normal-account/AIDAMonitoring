#!/usr/bin/env bash

if [[ -z "$POSTGRESQL" ]]
then
  echo "Error: variable POSTGRESQL is not set. run the env file in bin directory first"
  exit 1
fi

for i in {1..8}; do
	./run_benchmark.sh tpch > null$i.txt &
	sleep 1
done
