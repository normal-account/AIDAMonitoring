pgrep -f intermittent | xargs kill -9 2> /dev/null

sleep 1

for i in $(seq 1 $CLIENTS); do
        ./intermittent_burn_cpu $i &
done