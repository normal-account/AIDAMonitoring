psql -U bixi bixi -c "SELECT pid, wait_event_type, wait_event, state, query FROM pg_stat_activity WHERE wait_event IS NOT NULL";
psql -U bixi bixi -c "SELECT pid, relation::regclass, mode, granted FROM pg_locks WHERE NOT granted";
