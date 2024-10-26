import pandas as pd
from typing import Union, Optional, Any
from collections import namedtuple
from satoriengine.framework.pipelines.interface import PipelineInterface, TrainingResult


class StarterPipeline(PipelineInterface):
    @staticmethod
    def train(**kwargs) -> TrainingResult:
        if kwargs["stable"] is None:
            status, model = StarterPipeline.starterEnginePipeline(kwargs["data"])
            if status == 1:
                return TrainingResult(status, model, False)
        else:
            return TrainingResult(1, kwargs["stable"], True)

    @staticmethod
    def save(model: Optional[Any], modelpath: str) -> bool:
        """saves the stable model to disk"""
        return True

    @staticmethod
    def compare(
        stable: Optional[Any] = None,
        pilot: Optional[Any] = None,
    ) -> bool:
        """true indicates the pilot model is better than the stable model"""
        if stable is None:
            return True
        return pilot[0].backtest_error < stable[0].backtest_error

    @staticmethod
    def predict(**kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        status, predictor_model = StarterPipeline.starterEnginePipeline(kwargs["data"])
        if status == 1:
            return predictor_model[0].forecast
        return None

    @staticmethod
    def starterEnginePipeline(starter_dataset: pd.DataFrame):
        """Starter Engine function for the Satori Engine"""

        Result = namedtuple(
            "Result",
            ["forecast", "backtest_error", "model_name", "unfitted_forecaster"],
        )

        if len(starter_dataset) == 1:
            # If dataset has only 1 row, return the same value in the forecast dataframe
            value = starter_dataset.iloc[0, 1]
            forecast = pd.DataFrame(
                {"ds": [pd.Timestamp.now() + pd.Timedelta(days=1)], "pred": [value]}
            )
        elif len(starter_dataset) == 2:
            # If dataset has 2 rows, return their average
            value = starter_dataset.iloc[:, 1].mean()
            forecast = pd.DataFrame(
                {"ds": [pd.Timestamp.now() + pd.Timedelta(days=1)], "pred": [value]}
            )

        starter_result = Result(
            forecast=forecast,
            backtest_error=20,
            model_name="starter_dataset_model",
            unfitted_forecaster=None,
        )

        return 1, [starter_result]
