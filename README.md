# Creating the PostgreSQL container:

The container can be launched using the docker-compose.yml file. Simply run the following command:

```bash
docker compose up -d --build
```

Which will build and run the container (it might take a while).

Then, attach to the container :

```bash
docker container exec -it postgres_aida /bin/bash
```


# Running Benchbase benchmarks

Travel to the benchbase folder :

```bash
cd /home/build/benchbase
```

From there, a script is available (`run_benchmark.sh`). Run it to launch a benchmark:

```bash
./run_benchmark.sh <benchmark_name>
```

To display the list of available benchmarks, you can also run the script without arguments.

Configuration files for each benchmark can be found under `config/postgres`. 

See the benchbase GitHub page for more details: https://github.com/cmu-db/benchbase



# Disabling statistics in Postgres

Statistics are enabled by defaut. To disable them, first open the Postgres configuration file:

```bash
cd /home/build/postgres/
vim pg_storeddata/postgresql.conf
```

Scroll down to the `STATISTICS` section and disable all options by turning them to `off` or `none`.

Alternatively, you can simply overwrite the default configuration file with the example configuration files copied with this project:

```bash
mv postgresql_stats_on.conf pg_storeddata/postgresql.conf  # To enable stats (default)
# or
mv postgresql_stats_off.conf pg_storeddata/postgresql.conf # To disable stats
```

Once the modifications have been made, reload the postgres config:

```sql
psql -U admin -d benchbase
SELECT pg_reload_conf();
```

If you want to restart the Postgres server instead of dynamically reloading the config, you can run the `./start_db.sh` script:

```bash
/home/build/postgres/start_db.sh
```

You can confirm the configuration file has been loaded correctly by displaying the modified configurations via psql:

```sql
psql -U admin -d benchbase
SHOW track_activities
SHOW track_counts
```

And so on. 



# Running AIDA

AIDA is not necessary for running benchmarks. You can still set it up for test purposes.

Go to the AIDA Postgres folder:

```bash
cd /home/build/AIDA/aidaPostgreSQL/scripts
```

Run the `env.sh` script to set the needed environment variables:

```bash
. env.sh
```

Then, run the setup script:

```bash
./setup_postgresql.sh <db> <user>
```

And finally run the startup script:

```bash
./startup_postgresql.sh <db> <user>
```

If you want to run AIDA benchmarks, you will need to place the `postgres_data.zip` archive at the root of the project directory. It's currently not in the repository because of its large size.  