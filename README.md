# Creating the PostgreSQL container:

The container can be launched using the docker-compose.yml file. Simply run the following command:

```bash
docker compose up -d --build
```

Which will build and run the container (it might take a while).

Then, attach to the container :

```bash
docker container exec -it postgres_benchbase /bin/bash
```

**User**: `aida-user`
**Password**: `aida`


# 1. Running Benchbase benchmark

Travel to the Benchbase folder :

```bash
cd /home/build/benchbase
```

To run 4 clients of YCSB without competing UDFs, you can run:

```
sudo cgexec -g "cpu:parent/hw" ./run_benchmark.sh ycsb
```

Then, if you want to start the 4 competing UDFs, you can run the following command from another terminal:

```
./startup_udf_ycsb_bench.sh
```

The number of Benchbase clients is specified in the `config/postgres/sample_ycsb_config.xml` file.
The number of UDF clients is specified by the `CLIENTS` env var (default value is 4).

If you change the number of clients, you'll also need to rerun the cgroup script:

```
./create_cgroups.sh # This will recreate the cgroups (and update cpuset.cpus)
```


# 2. Running intermittent burn benchmark

From the `/home/build/` folder, there's a Makefile which compiled `burn_cpu` and `intermittent_burn_cpu` during the Docker build.


To run 4 clients of `intermittent_burn_cpu` without competition, run:

```
./run_intermittent_bench.sh
``` 

To run these 4 clients of both executables against each other, simply run:

```
./run_burn_compare_bench.sh
```


Note that `intermittent_burn_cpu` runs for 30 seconds by default. You'll need to recompile it to change that.