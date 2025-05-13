
if [ $# -ne 1 ]; then
	echo "./gen_folded.sh <name>"
	exit 1
fi

#while [ "$(pgrep -fc benchbase)" -lt 1 ]; do
#    sudo sleep 1
#done

echo Starting...

./get_last_pids.sh

sudo perf sched record -a -- sleep 120
sudo perf sched timehist > $1.txt
