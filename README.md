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

## Container credentials 


**User**: `aida-user`


**Password**: `aida`


## Database credentials

**Database name**: `benchbase`


**Database user**: `admin`


**Database password**: `password`

## Restarting the DB

To restart the DB at any point (and kill any UDFs), you can run:

```
/home/build/postgres/start_db.sh
```



# 1. Running the Benchbase benchmark

Travel to the Benchbase folder :

```bash
cd /home/build/benchbase
```

To run 4 clients of YCSB without competing UDFs, you can run:

```
./run_benchmark.sh ycsb
```

**Then**, if you want to start the 4 competing UDFs, you can run the following command from another terminal afterwards:

```
./startup_udf_ycsb_bench.sh
```

The number of Benchbase clients is specified in the `config/postgres/sample_ycsb_config.xml` file.

The number of UDF clients is specified by the `CLIENTS` env var (default value is 4).

If change the number of UDF clients, you'll also need to rerun the cgroup script:

```
./create_cgroups.sh # This will recreate the cgroups (and update cpuset.cpus)
```



# 2. Running the intermittent burn benchmark

From the `/home/build/` folder, there's a Makefile which compiled `burn_cpu` and `intermittent_burn_cpu` during the Docker build.


To run 4 clients of `intermittent_burn_cpu` without competition, run:

```
./run_intermittent_bench.sh
``` 

**Or**, to run these 4 clients of both executables against each other, you can instead run:

```
./run_burn_compare_bench.sh
```


Note that `intermittent_burn_cpu` runs for 30 seconds by default. You'll need to recompile it to change that.


# 3. Running the custom SQL-query benchmark

The custom benchmark is a simple C program which sends SQL requests continuously. Before sending a new query, it waits for the result from the previous one, like a normal application would.

To run multiple clients of the custom benchmark, run the following script in `/home/build/benchbase`:

```
./run_custom_bench.sh
```

The number of clients depends on your `CLIENTS` environment variable. The default is 4.

To start the competing UDFs and assign all processes to their respective cgroups, you can then run:

```
./startup_udf_ycsb_bench.sh
```

Note that this command will fail if you haven't run the benchbase YCSB benchmark once (`./run_benchmark ycsb`) as seen above, as the `usertable` table needs to be created first.


This benchmark does not print its number of iterations, but you can see the drastic CPU drop using `htop` when the competing UDFs are started.