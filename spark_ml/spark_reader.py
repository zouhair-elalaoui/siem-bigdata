import os
os.environ['PYSPARK_SUBMIT_ARGS'] = (
    '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 '
    'pyspark-shell'
)

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SIEM-Reader") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "flux-reseau") \
    .option("startingOffsets", "earliest") \
    .load()

messages = df.selectExpr(
    "CAST(key AS STRING) as key",
    "CAST(value AS STRING) as value",
    "timestamp"
)

query = messages \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

print("Spark lit Kafka — topic: flux-reseau")
query.awaitTermination()
