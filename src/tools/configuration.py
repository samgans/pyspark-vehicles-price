import os
from typing import Optional, Tuple


class ETLJobConfiguration:
    """
    Pretty obvious confguration created to reduce the dependency of the
    processors and modelling code on the code of the main module.
    """

    BASE_DIR: str = os.path.dirname(__file__)
    IS_DEV: bool = (os.environ.get("DEBUG") == "True")
    if IS_DEV:
        DATA_PATH: Optional[str] = os.path.dirname(BASE_DIR) + "/data/cars_8k.csv" 
    else:
        DATA_PATH = os.environ.get("DATA_PATH")
    LABEL_COL: str = "selling_price"
    CATEGORICAL_COLUMNS: Tuple[str] = ("fuel", "seller_type", "transmission")
