from functools import partial
from typing import (
    Any, Dict, Literal, List,
    Sequence, Tuple, Union
)
from pyspark.ml.base import _FitMultipleIterator, Estimator, Transformer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.param import Param
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame


eval_metric = Union[Literal["r2"], Literal["mae"],
                    Literal["rmse"], Literal["mse"]]
params_dict = Dict[str, Any]
model_params = Sequence[Tuple[Param, List[Union[float, int]]]]
threadsafe_iter = _FitMultipleIterator


def generate_linreg_parameters(lr: LinearRegression) -> model_params:
    elasticnet_params = [x / 10 for x in range(11)]
    regularization_params = [0.0001]

    reg_param = 0.0001
    while reg_param < 25:
        reg_param *= 2
        regularization_params.append(reg_param)
    params = ((lr.elasticNetParam, elasticnet_params),
              (lr.regParam, regularization_params))
    return params


def generate_gbt_params(gbt: GBTRegressor) -> model_params:
    depth_vals = [5]
    instances_in_node = [30]
    return (
        (gbt.maxDepth, depth_vals),
        (gbt.minInstancesPerNode, instances_in_node)
    )


def cross_validate_reg(train_set: DataFrame, param_vals: model_params,
                       estimator: Estimator, label_col: str,
                       prediction_col: str = "prediction",
                       metric_name: str = "mse") -> Transformer:
    reg_eval = RegressionEvaluator(
        predictionCol=prediction_col,
        labelCol=label_col,
        metricName=metric_name
    )

    gridbuilder = ParamGridBuilder()
    for values in param_vals:
        gridbuilder = gridbuilder.addGrid(values[0], values[1])
    grid = gridbuilder.build()
    crossval = CrossValidator(estimator=estimator, estimatorParamMaps=grid,
                              evaluator=reg_eval, numFolds=3)
    return crossval.fit(train_set)


def evaluate_regression(predictions: DataFrame, label_col: str, prediction_col: str,
                        metric_name: eval_metric = "r2", **kwargs) -> float:
    lr_evaluator = RegressionEvaluator(
        predictionCol=prediction_col,
        labelCol=label_col,
        metricName=metric_name,
        **kwargs
    )
    return lr_evaluator.evaluate(predictions)


def apply_modelling(dataset: DataFrame, label_col: str,
                    features_col: str, split: List[float]) \
                                            -> Tuple[DataFrame, int, int, int]:
    train_set, test_set = dataset.randomSplit(split)
    model = GBTRegressor(labelCol=label_col, featuresCol=features_col)
    model_params = generate_gbt_params(model)
    cv_best = cross_validate_reg(train_set, model_params,
                                 model, label_col=label_col)

    fitted = cv_best.transform(test_set)
    evaluator = partial(
        evaluate_regression,
        predictions=fitted,
        label_col=label_col,
        prediction_col="prediction"
    )
    best_model = cv_best.bestModel

    m_depth = best_model._java_obj.getMaxDepth()
    min_inst = best_model._java_obj.getMinInstancesPerNode()
    r2 = evaluator()
    mse = evaluator(metric_name="mse")
    rmse = evaluator(metric_name="rmse")

    model_metrics = {
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "min_depth": m_depth,
        "min_instances": min_inst
    }
    return (fitted, model_metrics)
