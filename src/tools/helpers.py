from py4j.java_gateway import JavaObject
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType


class SparkJVMLogger:
    """
    A wrapper for Spark Log4j logger.

    Created to make type hinting easier and move logger retrieving from
    the session creation logic.
    """

    @classmethod
    def getLogger(cls, sc: SparkSession, name: str) -> "SparkJVMLogger":
        java_logger = sc._jvm.org.apache.log4j.LogManager.getLogger(name)
        return cls(java_logger)

    def __init__(self, logger: JavaObject):
        self.logger = logger

    def error(self, message: str) -> None:
        self.logger.error(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warn(self, message: str) -> None:
        self.logger.warn(message)


def create_session(is_dev: bool) -> SparkSession:
    session = SparkSession.builder.appName("CarPriceAnalysis")
    if is_dev:
        session = session.master("local[*]")
    sc = session.getOrCreate()
    return sc


def read_data(sc: SparkSession, path: str, data_model: StructType,
              **kwargs) -> DataFrame:
    """Take the data path and read it."""
    return sc.read.schema(data_model).csv(path, **kwargs)
