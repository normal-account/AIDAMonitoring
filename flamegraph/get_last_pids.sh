KEYWORD_BENCH="bixi bixi 127.0.0.1\("
KEYWORD_AIDA="bixi bixi \[local\]"

echo > last_pids_bench.txt
pgrep -f "$KEYWORD_BENCH" | while read pid; do echo $pid >> last_pids_bench.txt; done

echo > last_pids_aida.txt
pgrep -f "$KEYWORD_AIDA" | while read pid; do echo $pid >> last_pids_aida.txt; done
