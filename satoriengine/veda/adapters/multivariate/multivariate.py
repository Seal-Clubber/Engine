import pandas as pd
from typing import Union
from collections import namedtuple
from sklearn.linear_model import LinearRegression
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult


class MultivariateAdapter(ModelAdapter):

    @staticmethod
    def condition(*args, **kwargs) -> float:
        if (
            isinstance(kwargs.get('availableRamGigs'), float)
            and kwargs.get('availableRamGigs') < .1
        ):
            return 1.0
        if len(kwargs.get('data', [])) < 10:
            return 1.0
        return 0.0

    def __init__(self, **kwargs):
        super().__init__()
        self.model = None

    def load(self, modelPath: str, **kwargs) -> Union[None, "ModelAdapter"]:
        """loads the model model from disk if present"""
        # copy over

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        # copy over
        return True

    def fit(self, data: pd.DataFrame, **kwargs) -> TrainingResult:
        # fill in
        return TrainingResult(-1, self)

    def compare(self, other: ModelAdapter, **kwargs) -> bool:
        # fill in
        return True

    def score(self, **kwargs) -> float:
        # fill in
        return 0.0

    def predict(self, data, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        # fill in
        return 0
