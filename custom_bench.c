#include <stdio.h>
#include <stdlib.h>
#include <libpq-fe.h>

int main() {
    const char *conninfo = "host=127.0.0.1 dbname=benchbase user=admin";
    PGconn *conn = PQconnectdb(conninfo);

    if (PQstatus(conn) != CONNECTION_OK) {
        fprintf(stderr, "Connection failed: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        return 1;
    }

    // Set non-blocking mode
    if (PQsetnonblocking(conn, 1) != 0) {
        fprintf(stderr, "Failed to set non-blocking mode: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        return 1;
    }

    while (1) {

       int id = (rand() % 10000) + 1;
       char query[128];
       snprintf(query, sizeof(query), "SELECT * FROM usertable WHERE ycsb_key = %d;", id);

       if (PQsendQuery(conn, query) == 0) {
                fprintf(stderr, "Send failed: %s\n", PQerrorMessage(conn));
                break;
       }
        // Drain all results (non-blocking)

        //if ( PQconsumeInput(conn) ) {
            while (1) {
                PGresult *res = PQgetResult(conn);
                if (!res) break;

                PQclear(res);
            }
        //}
    }

    PQfinish(conn);
    return 0;
}