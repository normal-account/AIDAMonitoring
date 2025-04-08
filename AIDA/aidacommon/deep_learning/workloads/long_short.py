#!/bin/bash

NAME="long"
python ../read_from_db_no_iter.py $NAME 0.73&
#python ../read_from_db_no_iter.py $NAME 5&
sleep 3

TASK="nn"
for ((i=1;i <= 3 ;i++))
do
        NAME="$TASK$i"
        echo "$NAME"
	python ../read_*Swi* $NAME 180000 18000&
        sleep 3
done



NAME="short"
python ../read_from_db_no_iter.py $NAME 0.73&
#python ../read_from_db_no_iter.py $NAME 5&
