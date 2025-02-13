# TPCH

We provide a comparison of the TPC-H Benchmark with:
* MonetDB (SQL)
* AIDA (Tabular objects API)
* pandas (DataFrame API)

## Requirements

* Python 3+
* [pandas](https://pandas.pydata.org/)
* [pymonetdb](https://github.com/gijzelaerr/pymonetdb)
* [MonetDB](https://www.monetdb.org/Home)
* [AIDA](https://github.com/joedsilva/AIDA)

## Configuration

Several parameters can be set in the files `TPCHconfig-XX.py`.
* `host`: node where the MonetDB database (or AIDA server) is running
* `dbname`: name of the database
* `schema`: (_MonetDB and pandas only_) name of the schema
* `user`: name to use to connect to the database
* `passwd`: password to use to connect to the database
* `databuffersize`: (_MonetDB and pandas only_) buffer size to fetch data from the database
* `SF`: scale factor of the TPCH dataset
* `preload`: (_pandas only_) if `True`, all tables are preloaded into memory before any query execution and the time it took is displayed
* `outputDir`: path to the folder where to output the results, it is created if not already existing
* `jobName`: (_AIDA only_) name to give to the job
* `port`: (_AIDA only_) port of the AIDA server
* `udfVSvtable`: (_AIDA only_) the tables are loaded as NumPy arrays, and the queries are executed with either table UDFs or virtual tables depending on the AIDA configuration

## Running

To run all the queries, simply execute the following:

```Bash
python3 runTPCH-MonetDB.py {1..22}
python3 runTPCH-AIDA.py {1..20} 22
python3 runTPCH-pandas.py {1..22}
```

The result of each query is displayed, as well as the time it took to be executed.

## Output

Each line of the output file `time-XX.csv` is organized as follows:

```
<query_number>,<time_in_seconds>
```

The file is not reset if already existing, new lines get appended.
