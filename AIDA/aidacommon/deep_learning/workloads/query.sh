#!/bin/bash


NAME="CPU"
python ../read_*Swi* $NAME 1000 40000&

sleep 4
NAME="GPU"
python ../read_*Swi* $NAME 180000 18000&

sleep 8
python ../../short_query.py

sleep 3
python ../../query.py
