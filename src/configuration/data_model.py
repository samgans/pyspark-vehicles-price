from pyspark.sql.types import (
    IntegerType, StructField, StringType, StructType
)


Vehicle = StructType([
    StructField("name", StringType(), False),
    StructField("year", IntegerType(), False),
    StructField("selling_price", IntegerType(), False),
    StructField("km_driven", IntegerType(), False),
    StructField("fuel", StringType(), False),
    StructField("seller_type", StringType(), False),
    StructField("transmission", StringType(), False),
    StructField("owner", StringType(), False),
    StructField("mileage", StringType(), False),
    StructField("engine", StringType(), False),
    StructField("max_power", StringType(), False),
    StructField("torque", StringType(), False),
    StructField("seats", IntegerType(), True)
])
