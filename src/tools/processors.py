from typing import Tuple

import pyspark.sql.functions as fnc
from pyspark.ml import Pipeline
from pyspark.ml.base import Estimator
from pyspark.ml.feature import (
    Normalizer, OneHotEncoder,
    StringIndexer, VectorAssembler
)
from pyspark.sql import DataFrame


def remove_unneeded(dataset: DataFrame) -> DataFrame:
    return dataset.drop("name", "engine", "max_power", "mileage", "torque")


def get_owner_number(dataset: DataFrame) -> DataFrame:
    processed = dataset.withColumn(
        "num_owners",
        fnc.when((fnc.col("owner") == "First Owner"), 1)
        .when((fnc.col("owner") == "Second Owner"), 2)
        .when((fnc.col("owner") == "Third Owner"), 3)
        .when((fnc.col("owner") == "Fourth & Above Owner"), 4)
    ).drop("owner")
    return processed


def count_years(dataset: DataFrame) -> DataFrame:
    formula = 2020 - fnc.col("year")
    return dataset.withColumn("years", formula).drop("year")


def create_processing_pipeline(columns: Tuple[str], cat_cols: Tuple[str],
                               features_col: str) -> Estimator:
    indexer_cols = tuple(f"{col}_ind" for col in cat_cols)
    encoder_cols = tuple(f"{col}_enc" for col in cat_cols)
    output_cols = [name for name in columns
                   if name not in cat_cols and name != "selling_price"]
    output_cols.extend(encoder_cols)

    indexer = StringIndexer(inputCols=cat_cols, outputCols=indexer_cols)
    encoder = OneHotEncoder(inputCols=indexer_cols, outputCols=encoder_cols)
    vectorizer = VectorAssembler(inputCols=encoder_cols, outputCol=features_col)
    normalizer = Normalizer(inputCol=features_col, outputCol=f"{features_col}_norm")
    return Pipeline(stages=[indexer, encoder, vectorizer, normalizer])


def process_vehicles_dataset(dataset: DataFrame, cat_cols: Tuple[str],
                             label_col: str, features_col: str) -> DataFrame:
    clean = remove_unneeded(dataset)
    years_counted = count_years(clean)
    owners_counted = get_owner_number(years_counted)
    owners_counted = owners_counted.na.drop("any")
    columns = owners_counted.schema.names
    pipeline = create_processing_pipeline(
        columns,
        cat_cols,
        features_col=features_col
    )
    processed = pipeline.fit(owners_counted).transform(owners_counted) \
        .withColumn("id", fnc.monotonically_increasing_id()) \
        .drop(features_col) \
        .withColumnRenamed(f"{features_col}_norm", features_col) \
        .select("id", features_col, label_col)
    return processed
