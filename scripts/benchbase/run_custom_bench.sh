echo Killing existing instances of custom_bench
pgrep custom_bench | xargs kill -9 2> /dev/null

echo Killing created UDFs...
pgrep -f "\[local\]" | xargs kill -9 2> /dev/null

sleep 1

for i in $(seq 1 $CLIENTS); do
        /home/build/custom_bench &
done