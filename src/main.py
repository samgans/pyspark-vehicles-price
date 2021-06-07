import os
import logging
from typing import Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

from data_model import Vehicle
from tools.helpers import SparkJVMLogger, create_session, read_data
from tools.modelling import apply_modelling
from tools.processors import process_vehicles_dataset


BASE_DIR: str = os.path.dirname(__file__)
IS_DEV: bool = (os.environ.get("DEBUG") == "True")
DATA_PATH: Optional[str] = os.path.dirname(BASE_DIR) + "/data/cars_8k.csv" \
                                if IS_DEV else os.environ.get("DATA_PATH")
LABEL_COL: str = "selling_price"
FEATURES_COL: str = "features"
CATEGORICAL_COLUMNS: Tuple[str] = ("fuel", "seller_type", "transmission")


def main() -> None:
    spark = create_session(IS_DEV)
    raw_data = read_data(spark, DATA_PATH, Vehicle, header=True)
    processed = process_vehicles_dataset(
        raw_data,
        cat_cols=CATEGORICAL_COLUMNS,
        label_col=LABEL_COL,
        features_col=FEATURES_COL
    )
    train_ds, test_ds = processed.randomSplit([0.6, 0.4])
    predictions = apply_modelling(
        train_ds,
        test_ds,
        label_col=LABEL_COL,
        features_col=FEATURES_COL
    )
    predictions.show(50)
    spark.stop()


if __name__ == "__main__":
    main()
