#!/usr/bin/env bash

dbhost=cerberus
user=test01
db=test01

table2=trand100x1r

warmup=5
repeat=100
numiters=5

for table in trand100x1r trand100x10r trand100x100r trand100x1000r trand100x10000r trand100x100000r trand100x1000000r
do
        cnt=1
        while [[ $cnt -le $numiters ]]
        do
            echo "SELECT 'MATMUL', '$table', '$table2', CAST(timee AS DECIMAL(30,20)) FROM la_op2('$table', '$table2', $warmup, $repeat);"
            let "cnt=$cnt+1"
        done |  $MONETDB/bin/mclient -u $user -d $db -h $dbhost
done