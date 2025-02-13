#!/usr/bin/env bash

dbhost=cerberus
user=test01
db=test01

warmup=5
repeat=10

for table in trand100x1r trand100x10r trand100x100r trand100x1000r trand100x10000r trand100x100000r trand100x1000000r
do
    cnt=1
    while [[ $cnt -le $warmup ]]
    do
        echo "SELECT '$table', * FROM load_data('$table');"
        let "cnt=$cnt+1"
    done |  $MONETDB/bin/mclient -u $user -d $db -h $dbhost >/dev/null

    cnt=1
    while [[ $cnt -le $repeat ]]
    do
        echo "SELECT 'DATA_LOAD', $cnt, '$table', * FROM load_data('$table');"
        let "cnt=$cnt+1"
    done |  $MONETDB/bin/mclient -u $user -d $db -h $dbhost
done
