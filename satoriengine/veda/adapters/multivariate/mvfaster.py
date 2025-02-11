import pandas as pd
import numpy as np
from typing import Union
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult
from autogluon.timeseries import TimeSeriesPredictor
from satorilib.logging import info, debug, warning
from satoriengine.veda.adapters.multivariate.mvpreprocess import conformData, createTrainTest, getSamplingFreq


class MultivariateFastAdapter(ModelAdapter):

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
        self.model: Union[MultivariateFastAdapter, None] = None
        self.dataTrain: pd.DataFrame = pd.DataFrame()
        self.dataTrainTest: pd.DataFrame = pd.DataFrame()
        self.modelError: float = 0
        self.covariateColNames: list[str] = []
        self.forecastingSteps: int = 1
        self.hyperparameters: dict = None # TODO : confirm if needed
        self.rng = np.random.default_rng(37)

    # TODO : have to confirm how model is going to be loaded, since its part of autogluon training
    def load(self, modelPath: str, **kwargs) -> Union[None, 'MultivariateFastAdapter']:
        """loads the model model from disk if present"""
        pass

    # TODO : have to confirm how model is going to be saved, since its part of autogluon training
    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        return True

    def fit(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> TrainingResult:
        self._manageData(targetData, covariateData)
        self.model = self._multivariateFit()
        return TrainingResult(1, self)

    def compare(self, other: Union['MultivariateFastAdapter', None] = None, **kwargs) -> bool:
        if not isinstance(other, self.__class__):
            return True
        thisScore = self.score()
        #otherScore = other.score(test_x=self.testX, test_y=self.testY)
        otherScore = other.modelError or other.score()
        isImproved = thisScore < otherScore
        if isImproved:
            info(
                'model improved!'
                f'\n  stable score: {otherScore}'
                f'\n  pilot  score: {thisScore}'
                f'\n  Parameters: {self.hyperparameters}',
                color='green')
        else:
            debug(
                f'\nstable score: {otherScore}'
                f'\npilot  score: {thisScore}')
        return isImproved

    def score(self, **kwargs) -> float:
        if self.model is None:
            return np.inf
        self.modelError = self.model.evaluate(self.dataTrain).get('MASE')*-1
        return self.modelError

    def predict(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        self._manageData(targetData, covariateData)
        predictionsFast = self.model.predict(self.dataTrain, known_covariates=self.dataTrainTest.drop('value', axis=1))
        resultDf = self._getPredictionDataframe(targetData, predictionsFast.mean()[0])
        return resultDf
    
    def _manageData(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame]):
        conformedData, self.covariateColNames = conformData(targetData, covariateData) 
        self.dataTrain, self.dataTrainTest = createTrainTest(conformedData, self.forecastingSteps)

    # TODO : should this be a property, maybe not because of the heavy computation?
    def _multivariateFit(self):
        return TimeSeriesPredictor(
                        prediction_length=self.forecastingSteps,
                        eval_metric="MASE",
                        target="value", 
                        known_covariates_names=self.covariateColNames, 
                        # log_file_path = log_file_path
                    ).fit(
                        self.dataTrain,
                        random_seed=self.rng,
                        hyperparameters={
                            "Naive": {}, 
                            "SeasonalNaive": {}, 
                            "Average": {},
                            "SeasonalAverage": {},
                            "DirectTabular": {"ag_args": {"name_suffix": "WithLighGBMRegressor"},},
                            "RecursiveTabular": {"ag_args": {"name_suffix": "WithLighGBMRegressor"},},
                        },
                        num_val_windows = 7,
                        val_step_size = self.forecastingSteps,
                        time_limit=3600,
                        enable_ensemble=True,
                    )
    
    @staticmethod
    def _getPredictionDataframe(targetDataframe: pd.DataFrame, predictionValue: float):
        targetDataframe['date_time'] = pd.to_datetime(targetDataframe['date_time'])
        target_df = targetDataframe.set_index('date_time')
        samplingFrequency = getSamplingFreq(target_df)
        futureDates = pd.date_range(
            start=pd.Timestamp(target_df.index[-1]) + pd.Timedelta(samplingFrequency),
            periods=1,
            freq=samplingFrequency)
        return pd.DataFrame({'date_time': futureDates, 'pred': predictionValue})
    
    