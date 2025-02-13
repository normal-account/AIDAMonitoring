if [ $# -ne 1 ]; then
        echo -e "Usage : ./run_benchmark <benchmark_name>\n"
        echo Available benchmarks :
        ls -l config/postgres/ | grep -Eo '[_].*[_]' | grep -Eo '[^_]*' | xargs -I {} echo -e "\t- {}"
        exit 0
fi

java -jar benchbase-postgres/benchbase.jar -b $1 -c config/postgres/sample_$1_config.xml --create=true --load=true --execute=true