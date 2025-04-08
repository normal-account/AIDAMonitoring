#!/bin/bash

#change $sorted and $num to be corresponding the submission time and #job
#./input.sh 10000 1000

num=10
sorted=(1 81 156 241 334 418 495 577 690 803)
startInterval=()
startInterval+=(${sorted[0]})

echo ${#sorted[@]}
for ((i=1;i < ${#sorted[@]};i++))
do
	interval=$((sorted[$i]-sorted[$(($i-1))]))
        startInterval+=($interval)
done


for ((i=1;i <= $num;i++))
do
        python ../read_*Swi* nn1 1000 40000&
        sleep 4
        python ../read_*Swi* nn2 180000 18000&
        sleep 4
        python ../read*k* k1 500&
        sleep 4
        python ../read*k* k1 500&
        sleep 4
        python ../read*k* k1 500&
        sleep 4
        python ../read*k* k1 500&
        sleep ${startInterval[$(($i-1))]}
done

