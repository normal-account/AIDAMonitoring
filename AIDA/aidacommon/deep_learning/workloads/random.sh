#!/bin/bash

# calculation:
# a)python ../read_*Swi* $NAME 1000 40000& cpu 20s
# b)python ../read_*Swi* $NAME 180000 18000& gpu 30s
# c)python ../read_*Swi* $NAME 25000 60000& gpu 44s
# d)python ../read*k* k1 1000& gpu 1s
# maximum number of jobs :a+b+4d+14s in 15 min:  12.5 set
# 80% capacity of system allows: 10 set -> slightly fewer 20 set
# first find the randomized submission time&interval
# ./random.sh 

num=30
sec=$((14*60))
totalTime=0
startTime=()
startInterval=()
for ((i=1;i <= $num;i++))
do
	Time=$(($RANDOM%$sec))
	#echo $Time
	startTime+=( $Time )

done

readarray -t sorted < <(printf '%s\n' "${startTime[@]}" | sort -n)
printf '%s ' "${sorted[@]}"
printf '\n'
echo "Done"

startInterval+=(${sorted[0]})
for ((i=1;i < $num;i++))
do
	interval=$((${sorted[$i]}-${sorted[$(($i-1))]}))
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


