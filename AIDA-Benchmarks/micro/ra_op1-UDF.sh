#!/usr/bin/env bash

dbhost=cerberus
user=test01
db=test01

table1=tnrand10x1000000r
table2=t2nrand10x1000000r

warmup=2
repeat=5

for numjoincols in 1 2 3 4 5 6 7 8 9 10 11
do
            echo "SELECT 'JOIN', $numjoincols, CAST(timee AS DECIMAL(30,20)), numrows FROM ra_op2('$table1', '$table2', $numjoincols, $warmup, $repeat);" |  $MONETDB/bin/mclient -u $user -d $db -h $dbhost
done