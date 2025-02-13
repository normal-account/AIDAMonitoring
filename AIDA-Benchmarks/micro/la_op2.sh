#!/usr/bin/env bash

table2=trand100x1r
repeat=5
for table in trand100x1r trand100x10r trand100x100r trand100x1000r trand100x10000r trand100x100000r trand100x1000000r
do
    cnt=0
    while [[ $cnt -lt $repeat ]]
    do
        python3 la_op2.py NumPyOpt $table $table2
        python3 la_op2.py pandasOpt $table $table2
        python3 la_op2.py AIDA-Matrix $table $table2
        python3 la_op2.py AIDA-RWS $table $table2
        let "cnt = $cnt + 1"
    done
done