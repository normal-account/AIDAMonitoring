
if [ $# -ne 1 ]; then
	echo "./gen_folded.sh <name>"
	exit 1
fi

while [ "$(pgrep -fc benchbase)" -lt 1 ]; do
    sudo sleep 1
done

echo Starting...

sudo perf record -a --call-graph fp -F 99 -- sleep 120
#sudo perf record -a --call-graph dwarf -F 99 -p $(ps aux | grep 'stress' | grep -v 'grep' | awk '{print $2}' | head -n 2 | paste -sd ',' -) -g -- sleep 30
sudo perf script > $1.perf
./stackcollapse-perf.pl $1.perf > $1.folded
./flamegraph.pl $1.folded > $1.svg
