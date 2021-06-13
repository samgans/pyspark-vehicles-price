from configuration import Configuration
from configuration.data_model import Vehicle
from tools.helpers import SparkJVMLogger, create_session, read_data
from tools.modelling import apply_modelling
from tools.processors import process_vehicles_dataset


def main() -> None:
    spark = create_session(Configuration.IS_DEV)
    logger = SparkJVMLogger.getLogger(spark, __file__)

    logger.info(f"CONFIGURATION:\nDebug: {Configuration.IS_DEV}\n"
                f"Data: {Configuration.DATA_PATH}\n"
                f"Label column: {Configuration.LABEL_COL}")
    logger.info("Loading the data...")
    raw_data = read_data(spark, Configuration.DATA_PATH, Vehicle, header=True)
    logger.info("Data processing is started.")
    processed = process_vehicles_dataset(
        raw_data,
        cat_cols=Configuration.CATEGORICAL_COLUMNS,
        label_col=Configuration.LABEL_COL,
        features_col=Configuration.FEATURES_COL,
        spare_cols=Configuration.SPARE_COLUMNS
    )
    logger.info("Processing is finished. Going to the training stage.")
    predictions, metrics = apply_modelling(
        processed,
        label_col=Configuration.LABEL_COL,
        features_col=Configuration.FEATURES_COL,
        split=Configuration.TRAIN_TEST_SPLIT
    )
    predictions.show(50)
    logger.info(
        f"R2: {metrics['r2']}\nMSE: {metrics['mse']}\n"
        f"RMSE: {metrics['rmse']}\nMinimal Tree Depth: {metrics['min_depth']}\n"
        f"Minimal objects per node: {metrics['min_instances']}"
    )
    spark.stop()


if __name__ == "__main__":
    main()
