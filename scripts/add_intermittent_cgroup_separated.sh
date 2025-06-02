i=0

pgrep -f "intermittent" | while read pid; do i=$(($i+1)); echo hw$i : $pid; echo $pid | sudo tee /sys/fs/cgroup/parent/hw$i/cgroup.procs; done