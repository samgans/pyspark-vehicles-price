# PySpark ETM Job

## Description

This repository contains PySpark code that **Extracts** the data on used vehicles prices, **Transforms** it with fully configurable processing pipeline, and creates a predicting **Model** based on the data being processed. The pipeline is a result of my Spark and its Python API study, so it was fully indended to boost the instrument knowledge rather that to create a precise model. Therefore, the quality of the hyperparameters tuning is rather pure, but the system is pretty much configurable, so anyone can create a set of parameters inside the specific functions and feed it to cross validator to find the optimal values.

The decision to use GBTRegressor on the **Modelling** step instead of Linear Regression is made based on observation that the first model has better prediction results than pure LinReg.

Data source: [Vehicle dataset](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv)

## Usage

1. Put the needed environmental variables inside your environment or `.env` file (the list of needed variables is inside `env.example` file).
2. Activate virtual environment using `pipenv shell` or `pipenv install` (if the modules were not previously installed).

### Running the pipeline:

```bash
foo@bar: ./run_analysis.sh
```

### Testing:

```bash
foo@bar: ./run_analysis.sh -t
```

## Project Structure

Project structure is based on the existing [Pyspark Project Example Structure](https://github.com/AlexIoannides/pyspark-example-project), so all the modules inside `/src/` are automatically zipped during the start of the job and then sent as `--python-files` in `spark-submit`.

