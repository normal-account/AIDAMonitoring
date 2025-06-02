./kill_burns.sh

sleep 1

for i in $(seq 1 $CLIENTS); do
        ./burn_cpu 1 &
done