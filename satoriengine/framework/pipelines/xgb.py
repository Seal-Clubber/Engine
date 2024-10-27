from typing import Iterable, Union
import traceback
import os
import time
import copy
import pandas as pd
import numpy as np
from reactivex import merge
from reactivex.subject import BehaviorSubject
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from satorilib import logging
from satorilib.concepts import StreamId, StreamOverview
from satorilib.api.disk import Cached
from satorilib.api.interfaces.model import ModelMemoryApi
from satoriengine.concepts import HyperParameter
from satoriengine.model.pilot import PilotModel
from satoriengine.model.stable import StableModel

from satoriengine.model.chronos_adapter import ChronosAdapter
from satoriengine.model.ttm_adapter import TTMAdapter
from typing import Union, Optional, Any
from collections import namedtuple
from satoriengine.framework.pipelines.interface import PipelineInterface, TrainingResult
from satoriengine.utils import clean


class XGBRegressorPipeline(PipelineInterface):

    def __init__(self, useGPU: bool, hyperParameters: list[HyperParameter], xgbParams: list = []):
        self.model = XGBRegressor(eval_metric='mae')

    def fit(self, **kwargs) -> TrainingResult:
        ''' train model - in place '''
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(
            data, df_target, test_size=self.split or 0.2, shuffle=False)
        # using data to train the model
        self.model.fit(
            self.trainX,
            self.trainY,
            eval_set=[(self.trainX, self.trainY), (self.testX, self.testY)],
            verbose=False)
        return TrainingResult(1, self.model, True)

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        return True

    def compare(self, other: Union[PipelineInterface, None] = None, **kwargs) -> bool:
        """true indicates the pilot model is better than the stable model"""
        if isinstance(other, self.__class__):
            return True
        return self._score(self.model) < self._score(other)

    def predict(self, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        return self.model.predict(kwargs["data"])

    def _produceTrainingSet(self):
        df = self.testFeatureSet.copy()
        df = df.iloc[0:-1, :]
        df = df.replace([np.inf, -np.inf], np.nan)
        df.columns = clean.columnNames(df.columns)
        # df = df.reset_index(drop=True)
        # df = coerceAndFill(df)
        df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

        lookback_len = next(
            (param.test for param in self.hyperParameters if param.name == 'lookback_len'), 1)
        data = df.to_numpy(dtype=np.float64).flatten()
        data = np.concatenate(
            [np.zeros((lookback_len,), dtype=data.dtype), data])
        data = [
            data[i-lookback_len:i]
            for i in range(lookback_len, data.shape[0])]
        df = pd.DataFrame(data)

        df_target = self.target.iloc[0:df.shape[0], :]
        data = df_target.to_numpy(dtype=np.float64).flatten()
        df_target = pd.DataFrame(data)

    def _score(self, model: XGBRegressor, testX: Iterable, testY: Iterable):
        return mean_absolute_error(testY, model.predict(testX))
