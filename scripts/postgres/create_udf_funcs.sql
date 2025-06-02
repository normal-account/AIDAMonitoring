CREATE OR REPLACE FUNCTION continuous_tpch_q17()
RETURNS void AS $$
DECLARE
    result numeric;
BEGIN
    LOOP
        SELECT SUM(l_extendedprice) / 7.0 into result 
        FROM lineitem
        JOIN part ON p_partkey = l_partkey
        WHERE p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (
              SELECT 0.2 * AVG(l_quantity)
              FROM lineitem
              WHERE l_partkey = p_partkey
          );
          
        result := NULL;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION continuous_ycsb(start_id BIGINT DEFAULT 0, end_id BIGINT DEFAULT 100000)
RETURNS void AS $$
DECLARE
    i BIGINT;
    result RECORD;
BEGIN
    LOOP
      i := start_id;
      WHILE i < end_id LOOP
          -- Run the YCSB-style SELECT
          SELECT * INTO result FROM usertable WHERE ycsb_key = i;
          result := NULL;

          i := i + 1;
      END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;