import logging
import os
import unittest

from pyspark.sql import SparkSession

from src.configuration import Configuration
from src.configuration.data_model import Vehicle
from src.tools.helpers import create_session
from src.tools.processors import (
    count_years, create_vectorization_pipeline,
    get_owners_number, process_vehicles_dataset
)


TEST_DATA_PATH: str = os.path.dirname(__file__) + "/data/test_set.csv"


class PySparkProcessingTests(unittest.TestCase):

    @classmethod
    def supress_logging(cls) -> None:
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)

    @staticmethod
    def get_session() -> SparkSession:
        session = create_session(True)
        return session

    @classmethod
    def setUpClass(cls) -> None:
        cls.supress_logging()
        cls.spark = cls.get_session()
        cls.data = cls.spark.read.schema(Vehicle).csv(TEST_DATA_PATH, header=True) \
            .drop(*Configuration.SPARE_COLUMNS)

    def test_owners_number(self) -> None:
        num_mapping = {
            1: "First Owner",
            2: "Second Owner",
            3: "Third Owner",
            4: "Fourth & Above Owner",
            0: "Test Drive Car"
        }
        processed = get_owners_number(self.data).select("owner", "num_owners")
        for row in processed.collect():
            num_owners = row["num_owners"]
            self.assertEqual(row["owner"], num_mapping[num_owners])

    def test_years(self) -> None:
        processed = count_years(self.data).select("year", "years")
        for row in processed.collect():
            manufactured = row["year"]
            valid_year = 2020 - manufactured
            self.assertEqual(valid_year, row["years"])

    def test_vectorization_pipeline(self) -> None:
        col_names = self.data.schema.names
        indexer_cols = list(f"{col}_ind" for col in Configuration.CATEGORICAL_COLUMNS)
        encoder_cols = list(f"{col}_enc" for col in Configuration.CATEGORICAL_COLUMNS)
        output_cols = [name for name in col_names
                       if name not in Configuration.CATEGORICAL_COLUMNS
                       and name != Configuration.LABEL_COL]
        output_cols.extend(encoder_cols)

        pipeline = create_vectorization_pipeline(
            col_names,
            Configuration.CATEGORICAL_COLUMNS,
            features_col=Configuration.FEATURES_COL,
            label_col=Configuration.LABEL_COL
        )
        indexer, encoder, assembler, normalizer = pipeline.getStages()
        self.assertEqual(indexer.getInputCols(), list(Configuration.CATEGORICAL_COLUMNS))
        self.assertEqual(indexer.getOutputCols(), indexer_cols)

        self.assertEqual(encoder.getInputCols(), indexer_cols)
        self.assertEqual(encoder.getOutputCols(), encoder_cols)

        self.assertEqual(assembler.getInputCols(), output_cols)
        self.assertEqual(assembler.getOutputCol(), Configuration.FEATURES_COL)

        self.assertEqual(normalizer.getInputCol(), Configuration.FEATURES_COL)
        self.assertEqual(normalizer.getOutputCol(), f"{Configuration.FEATURES_COL}_norm")

    def test_whole_processing(self) -> None:
        processed_data = process_vehicles_dataset(
            self.data,
            Configuration.CATEGORICAL_COLUMNS,
            Configuration.LABEL_COL,
            Configuration.FEATURES_COL,
            Configuration.SPARE_COLUMNS
        )
        column_names = processed_data.schema.names
        self.assertTrue("id" in column_names)
        self.assertTrue(Configuration.FEATURES_COL in column_names)
        self.assertTrue(Configuration.LABEL_COL in column_names)


if __name__ == "__main__":
    unittest.main()
