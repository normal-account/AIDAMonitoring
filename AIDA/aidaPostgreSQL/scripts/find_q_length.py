import sys

f = open("./q_length.txt", "a")
with open("/home/build/postgres/pg_storeddata/aidas.log","r") as input_file:
    for line in input_file:
        if(line[0:18]=='INFO:root:q_length'):
            f.write(line[20:])

f.close()
