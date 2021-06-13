import os
from typing import List, Optional, Tuple


class Configuration:

    BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))
    IS_DEV: bool = (os.environ.get("DEBUG") == "True")
    DATA_PATH: Optional[str] = os.path.dirname(BASE_DIR) + "/data/cars_8k.csv" \
                               if IS_DEV else os.environ.get("DATA_PATH")
    LABEL_COL: str = "selling_price"
    FEATURES_COL: str = "features"
    CATEGORICAL_COLUMNS: Tuple[str] = ("fuel", "seller_type", "transmission")
    SPARE_COLUMNS: Tuple[str] = ("name", "engine", "max_power", "mileage", "torque")
    TRAIN_TEST_SPLIT: List[float] = [0.8, 0.2]
