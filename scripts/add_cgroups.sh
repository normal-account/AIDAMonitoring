echo Adding bench cgroup...
./add_bench_cgroup.sh

echo Adding tpch cgroup...
benchbase/startup_udf_ycsb.sh bixi bixi
sleep 1
./add_aida_cgroup.sh

echo Adding Java cgroup...
./add_java_cgroup.sh
