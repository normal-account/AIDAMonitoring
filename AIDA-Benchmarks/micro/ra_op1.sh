#!/usr/bin/env bash

table1=tnrand10x1000000r
table2=t2nrand10x1000000r
numjoincols=1

python3 ra_op1.py pandasOpt $table1 $table2 $numjoincols
python3 ra_op1.py AIDA $table1 $table2 $numjoincols
python3 ra_op1.py AIDA-Matrix $table1 $table2 $numjoincols