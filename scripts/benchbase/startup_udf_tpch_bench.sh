#!/usr/bin/env bash

if [[ -z "$POSTGRESQL" ]]
then
  echo "Error: variable POSTGRESQL is not set. run the env file in bin directory first"
  exit 1
fi

echo Starting UDFs...
for i in $(seq 1 $CLIENTS); do
	$POSTGRESQL/bin/psql -U admin -d benchbase -c "select * from continuous_tpch_q17()" &
done

echo Adding high-weight group...
./add_hw_cgroup.sh

echo Adding low-weight group...
./add_lw_cgroup.sh