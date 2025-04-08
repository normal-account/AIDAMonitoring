# Start the Postgres SQL server
if [[ -z "$PGDATADIR" ]]
then
  echo "Error: variable PGDATADIR is not set. run the env file in the current directory first ('. env.sh')"
  exit 1
fi

pg_ctl stop -D $PGDATADIR -m immediate
pg_ctl start -D $PGDATADIR -l $PGLOGDIR/postgres.log -o -i