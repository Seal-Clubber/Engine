import pandas as pd
import numpy as np
from typing import Union
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult
from satorilib.logging import info, debug
from satoriengine.veda.adapters.multivariate.mvadapters import FastMVAdapter, LightMVAdapter, HeavyMVAdapter 


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
        self.model: ModelAdapter = None

    def load(self, modelPath: str, **kwargs) -> Union[None, "ModelAdapter"]:
        """loads the model model from disk if present"""
        # TODO : confirm after jerome looks on auto-gloun model saving
        pass

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        # TODO : confirm after jerome looks on auto-gloun model saving
        pass

    def fit(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> TrainingResult:
        if self.model is None:
            self.model = FastMVAdapter()
            self.model.fit(targetData, covariateData)  
        else:
            if 'conditionToCheckIfVps':
                # Train on the MV Adapter, but should have light computations for VPS
                # hyper-parameters
                # weighted ensemble
                self.model = LightMVAdapter()
                self.model.fit()
            else:  
                # Train on the MV Adapter which is slower to train but more accurate, Heavy computations
                self.model = HeavyMVAdapter()
                self.model.fit()  
        return self.model

    def compare(self, other: ModelAdapter, **kwargs) -> bool:
        return self.model.compare(other)

    def score(self, **kwargs) -> float:
        pass

    def predict(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        return self.model.predict(targetData, covariateData)
