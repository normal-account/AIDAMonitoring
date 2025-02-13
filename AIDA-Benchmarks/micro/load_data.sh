#!/usr/bin/env bash

for table in trand100x1r trand100x10r trand100x100r trand100x1000r trand100x10000r trand100x100000r trand100x1000000r
do
    python3 load_data.py NumPy $table
    python3 load_data.py NumPyOpt $table
    python3 load_data.py pandas $table
    python3 load_data.py pandasOpt $table
    python3 load_data.py AIDA $table
    python3 load_data.py AIDA-Matrix $table
done