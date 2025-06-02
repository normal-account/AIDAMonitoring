if [ $# -eq 0 ]; then
	MYHOME=/home/build
else
	MYHOME=$1
fi

export PATH=$MYHOME/postgres/installdir/bin:$PATH
export POSTGRESQL=$MYHOME/postgres/installdir
export PGDATADIR=$MYHOME/postgres/pg_storeddata
export PGDATA=$MYHOME/postgres/pg_storeddata
export PGLOGDIR=$MYHOME/postgres/pg_log
export PYTHONPATH=$PYTHONPATH:$MYHOME/AIDA:$MYHOME/AIDA-Benchmarks
export AIDACONFIG=$MYHOME/AIDA/misc/aidaconfig-p.ini
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64/
export CUDA_VISIBLE_DEVICES=1,0
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export CLIENTS=4