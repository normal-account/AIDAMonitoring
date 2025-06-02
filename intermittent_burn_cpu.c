#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <stdbool.h>

#define BENCH_RUN_TIME_S  30    // total bench run time (in seconds)
#define BURST_INTERVAL_MS 2     // interval which separates our busy loops (in ms)
#define BURST_DURATION_MS 3     // how long the CPU bursts last (in ms)
#define NUM_THREADS 1           // increase for higher CPU %

volatile bool run_burst = false;
volatile unsigned long long x = 0;

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void* burn_cpu(void* arg) {
    while (1) {
        if (run_burst) {
            for (unsigned long i = 0; i < 1e9; i++) {
                if ( ( i % 1000 ) == 0 )
                {
                    x += 1; // increment every 1000 iterations
                }
                if (!run_burst) break;
            }
        } else {
            usleep(10); // sleep briefly to reduce CPU use
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        printf("Usage : ./run_intermittent_burn <client id>\n");
        return 1;
    }

    pthread_t threads[NUM_THREADS];

    // start burner threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, burn_cpu, NULL);
    }

    double start = now_seconds();
    double last_sample = start;
    double last_burst = start;
    double now = 0;
    double elapsed = 0;

    while (elapsed < 30) {
        now = now_seconds();
        elapsed = now - start;

        // trigger burst every BURST_INTERVAL_MS
        if ((now - last_burst) * 1000.0 >= BURST_INTERVAL_MS) {
            run_burst = true;
            usleep(BURST_DURATION_MS * 1000);  // run burst for short time
            run_burst = false;
            last_burst = now;
        }

        usleep(1000);  // small sleep to avoid spinning in main thread
    }

    int client_num = atoi( argv[1] );

    usleep( 10000 * client_num-1 ); // so that our clients print in order

    if ( client_num == 1 ) printf("\n");

    printf("Iterations for worker %s : %lldm\n", argv[1], x/1000);

    return 0;
}