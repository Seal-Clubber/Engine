import pandas as pd
from typing import Union
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult


class HeavyMVAdapter(ModelAdapter):

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
        self.model: ModelAdapter = None

    def load(self, modelPath: str, **kwargs) -> Union[None, "ModelAdapter"]:
        """loads the model model from disk if present"""
        pass

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        pass

    def fit(self, data: pd.DataFrame, **kwargs) -> TrainingResult:
        return self.model
    
    def compare(self, other: ModelAdapter, **kwargs) -> bool:
        return True

    def score(self, **kwargs) -> float:
        return 0.0

    def predict(self, data, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        return 0
