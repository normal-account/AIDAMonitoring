#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

void* burn_cpu(void* arg) {
    while (1) {
        // Keep the CPU busy doing calculations
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    if (!threads) {
        perror("malloc");
        return 1;
    }

    for (int i = 0; i < num_threads; ++i) {
        if (pthread_create(&threads[i], NULL, burn_cpu, NULL) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    // Let threads run indefinitely
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    return 0;
}

