#!/usr/bin/env bash

if [[ -z "$POSTGRESQL" ]]
then
  echo "Error: variable POSTGRESQL is not set. run the env file in bin directory first"
  exit 1
fi

for i in $(seq 1 $CLIENTS); do
	$POSTGRESQL/bin/psql -U admin -d benchbase -c "select * from continuous_ycsb()" &
done

./add_hw_cgroup.sh
./add_lw_cgroup.sh