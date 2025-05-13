
if [ $# -ne 1 ]; then
	echo "./gen_folded.sh <name>"
	exit 1
fi

while [ "$(pgrep -fc stress)" -lt 1 ]; do
    sudo sleep 1
done

echo Starting...

sudo perf sched record -a -- sleep 30
sudo perf sched timehist > $1.txt
