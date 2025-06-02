./difffolded.pl old.folded new.folded > diff.folded
./flamegraph.pl --title "Differential with Off-CPU" --color=io diff.folded > diff.svg

