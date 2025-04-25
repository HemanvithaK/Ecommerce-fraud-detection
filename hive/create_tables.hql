CREATE EXTERNAL TABLE IF NOT EXISTS transactions (
    transaction_id STRING,
    user_id STRING,
    product_id STRING,
    category STRING,
    amount DOUBLE,
    timestamp STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/transactions/';
