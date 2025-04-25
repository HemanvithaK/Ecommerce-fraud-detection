from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum

spark = SparkSession.builder.appName("EcommerceBatch").getOrCreate()

df = spark.read.csv("data/transactions.csv", header=True, inferSchema=True)

df.groupBy("category")\
  .agg(sum("amount").alias("total_sales"))\
  .show()

df.write.mode("overwrite").parquet("output/batch_results/")
spark.stop()
