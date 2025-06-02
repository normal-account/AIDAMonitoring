\set id random(1, 10000)
SELECT * FROM usertable WHERE ycsb_key = :id;
