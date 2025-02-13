#!/usr/bin/env bash

repeat=5
for table in trand100x1r trand100x10r trand100x100r trand100x1000r trand100x10000r trand100x100000r trand100x1000000r
do
    cnt=0
    while [[ $cnt -lt $repeat ]]
    do
        python3 la_op1.py NumPyOpt $table
        python3 la_op1.py pandasOpt $table
        python3 la_op1.py AIDA-Matrix $table
        python3 la_op1.py AIDA-RWS $table
        let "cnt = $cnt + 1"
    done
done