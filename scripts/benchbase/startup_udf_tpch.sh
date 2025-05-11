#!/usr/bin/env bash

db=$1
usr=$2

if [ -z "$db" ]
then
  echo "Error: usage $0 <db> [ <usr> ]"
  exit 1
fi

if [[ -z "$POSTGRESQL" ]]
then
  echo "Error: variable POSTGRESQL is not set. run the env file in bin directory first"
  exit 1
fi

for i in {1..4}; do
	if [ -z "$usr" ]
	then
    		$POSTGRESQL/bin/psql -d $db -c "select * from continuous_tpch_q17();" &
	else
 	   	$POSTGRESQL/bin/psql -U $usr -d $db -c "select * from continuous_tpch_q17()" &
	fi
done
