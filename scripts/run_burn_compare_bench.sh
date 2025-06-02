echo "Number of clients : $CLIENTS"

echo Adding burn cgroup...
./run_burn_bench.sh
./add_burn_cgroup.sh
#./add_burn_cgroup_separated.sh

echo Adding intermittent cgroup...
./run_intermittent_bench.sh
./add_intermittent_cgroup.sh
#./add_intermittent_cgroup_separated.sh