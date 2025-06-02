all: burn_cpu intermittent_burn_cpu custom_bench

burn_cpu:
	gcc -O3 -o burn_cpu burn_cpu.c

intermittent_burn_cpu:
	gcc -O3 -o intermittent_burn_cpu intermittent_burn_cpu.c

custom_bench:
	gcc -O3 -o custom_bench custom_bench.c -I/usr/include/postgresql -lpq