
if [ $# -ne 1 ]; then
	echo "./gen_folded.sh <name>"
	exit 1
fi

while [ "$(pgrep -fc benchbase)" -lt 16 ]; do
    sudo sleep 1
done

sleep 10
echo Starting...

sudo perf record -e cpu-clock,sched:sched_switch -F 99 -p $(ps aux | grep 'admin benchbase' | grep -v 'grep' | awk '{print $2}' | paste -sd ',' -) -g -- sleep 180
sudo perf script > $1.perf
./stackcollapse-perf.pl $1.perf > $1.folded
