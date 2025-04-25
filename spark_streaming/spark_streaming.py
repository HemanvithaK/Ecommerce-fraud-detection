from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType

schema = StructType() \
    .add("transaction_id", StringType()) \
    .add("user_id", StringType()) \
    .add("product_id", StringType()) \
    .add("category", StringType()) \
    .add("amount", DoubleType()) \
    .add("timestamp", StringType())

spark = SparkSession.builder.appName("EcommerceStreaming").getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .load()

value_df = df.selectExpr("CAST(value AS STRING)")
json_df = value_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

query = json_df.writeStream \
    .format("parquet") \
    .option("path", "output/stream_results/") \
    .option("checkpointLocation", "output/checkpoints/") \
    .outputMode("append") \
    .start()

query.awaitTermination()
