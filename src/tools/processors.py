from typing import Tuple

import pyspark.sql.functions as fnc
from pyspark.ml import Pipeline
from pyspark.ml.base import Estimator
from pyspark.ml.feature import (
    Normalizer, OneHotEncoder,
    StringIndexer, VectorAssembler
)
from pyspark.sql import DataFrame


def get_owners_number(dataset: DataFrame) -> DataFrame:
    processed = dataset.withColumn(
        "num_owners",
        fnc.when((fnc.col("owner") == "First Owner"), 1)
        .when((fnc.col("owner") == "Second Owner"), 2)
        .when((fnc.col("owner") == "Third Owner"), 3)
        .when((fnc.col("owner") == "Fourth & Above Owner"), 4)
        .when((fnc.col("owner") == "Test Drive Car"), 0)
    )
    return processed


def count_years(dataset: DataFrame) -> DataFrame:
    formula = 2020 - fnc.col("year")  # 2020 is a year of the target dataset
    return dataset.withColumn("years", formula)


def create_vectorization_pipeline(columns: Tuple[str], cat_cols: Tuple[str],
                                  features_col: str, label_col: str,
                                  norm: bool = True) -> Estimator:
    indexer_cols = tuple(f"{col}_ind" for col in cat_cols)
    encoder_cols = tuple(f"{col}_enc" for col in cat_cols)
    output_cols = [name for name in columns
                   if name not in cat_cols and name != label_col]
    output_cols.extend(encoder_cols)

    indexer = StringIndexer(inputCols=cat_cols, outputCols=indexer_cols)
    encoder = OneHotEncoder(inputCols=indexer_cols, outputCols=encoder_cols)
    vectorizer = VectorAssembler(inputCols=output_cols, outputCol=features_col)
    stages = [indexer, encoder, vectorizer]
    if norm:
        normalizer = Normalizer(inputCol=features_col, outputCol=f"{features_col}_norm")
        stages.append(normalizer)
    return Pipeline(stages=stages)


def process_vehicles_dataset(dataset: DataFrame, cat_cols: Tuple[str],
                             label_col: str, features_col: str,
                             spare_cols: Tuple[str]) -> DataFrame:
    clean = dataset.drop(*spare_cols)
    years_counted = count_years(clean).drop("year")
    owners_counted = get_owners_number(years_counted) \
        .drop("owner") \
        .na.drop("any")
    columns = owners_counted.schema.names
    pipeline = create_vectorization_pipeline(
        columns,
        cat_cols,
        features_col=features_col,
        label_col=label_col
    )
    processed = pipeline.fit(owners_counted).transform(owners_counted) \
        .select(""f"{features_col}_norm", label_col) \
        .withColumn("id", fnc.monotonically_increasing_id()) \
        .withColumnRenamed(f"{features_col}_norm", features_col)
    return processed
