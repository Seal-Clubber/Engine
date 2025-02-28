import pandas as pd
import numpy as np
from typing import Union
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from satorilib.logging import info, debug
from Engine.satoriengine.veda.adapters.multivariate.data import conformData, createTrainTest, getSamplingFreq


class FastMVAdapter(ModelAdapter):

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
        self.model: Union[FastMVAdapter, None] = None
        self.dataTrain: pd.DataFrame = pd.DataFrame()
        self.fullDataset: pd.DataFrame = pd.DataFrame()
        self.modelError: float = 0
        self.covariateColNames: list[str] = []
        self.forecastingSteps: int = 1
        self.rng = np.random.default_rng(37)

    # TODO : have to confirm how model is going to be loaded, since its part of autogluon training
    def load(self, modelPath: str, **kwargs) -> Union[None, 'FastMVAdapter']:
        """loads the model model from disk if present"""
        pass

    # TODO : have to confirm how model is going to be saved, since its part of autogluon training
    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        return True

    def fit(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> TrainingResult:
        self._manageData(targetData, covariateData)
        self.model = self._multivariateFit()
        # TODO : confirm about .refit stuff
        return TrainingResult(1, self)

    def compare(self, other: Union['FastMVAdapter', None] = None, **kwargs) -> bool:
        # if not isinstance(other, self.__class__):
        #     return True
        # thisScore = self.score()
        # #otherScore = other.score(test_x=self.testX, test_y=self.testY)
        # otherScore = other.modelError or other.score()
        # isImproved = thisScore < otherScore
        # if isImproved:
        #     info(
        #         'model improved!'
        #         f'\n  stable score: {otherScore}'
        #         f'\n  pilot  score: {thisScore}',
        #         color='green')
        # else:
        #     debug(
        #         f'\nstable score: {otherScore}'
        #         f'\npilot  score: {thisScore}')
        # return isImproved
        return True

    def score(self, **kwargs) -> float:
        # if self.model is None:
        #     return np.inf
        # self.modelError = self.model.evaluate(self.fullDataset).get('MASE')*-1
        # return self.modelError
        return 0.0

    def predict(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        self._manageData(targetData, covariateData)
        datasetWithFutureCov = self.appendCovariateFuture(self.fullDataset)
        prediction = self.model.predict(self.fullDataset, known_covariates=datasetWithFutureCov.drop('value', axis=1))
        resultDf = self._getPredictionDataframe(targetData, prediction.mean()[0]) # TODO: can also use in-built auto-gluon stuff ( optimize )
        return resultDf
    
    def _manageData(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame]):
        conformedData, self.covariateColNames = conformData(targetData, covariateData) 
        self.dataTrain, self.fullDataset = createTrainTest(conformedData, self.forecastingSteps)

    def _multivariateFit(self) -> TimeSeriesPredictor:
        return TimeSeriesPredictor(
                        prediction_length=self.forecastingSteps,
                        eval_metric="MASE",
                        target="value", 
                        known_covariates_names=self.covariateColNames, 
                        verbosity=0
                    ).fit(
                        self.fullDataset,
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
    def appendCovariateFuture(df: pd.DataFrame, covariateColNameList: list[str]) -> pd.DataFrame:
        for colName in covariateColNameList:
            covDf = df[[colName]]
            if not covDf.empty:
                predictor = TimeSeriesPredictor(
                    prediction_length=1,
                    eval_metric="MASE",
                    target=colName,
                    verbosity=0,
                ).fit(
                    covDf,
                    hyperparameters={
                        "RecursiveTabular": {"ag_args": {"name_suffix": "WithLighGBMRegressor"}}
                    },
                    num_val_windows=1,
                    val_step_size=1,
                    time_limit=600,
                    enable_ensemble=False,
                )
                prediction = predictor.predict(covDf)
                covariatePredictionValue = prediction.mean()[0]
                covariatePredictionTimestamp = str(prediction.index[0][1])
                new_row = pd.DataFrame({
                    'timeseriesid': [df.index[0][0]],
                    'date_time': [covariatePredictionTimestamp],
                    colName: [covariatePredictionValue]
                })
                new_row = TimeSeriesDataFrame.from_data_frame(
                    new_row,
                    id_column="timeseriesid",
                    timestamp_column="date_time"
                )
                df = pd.concat([df, new_row])
        return df
    
    @staticmethod
    def _getPredictionDataframe(targetDataframe: pd.DataFrame, predictionValue: float) -> pd.DataFrame:
        targetDataframe['date_time'] = pd.to_datetime(targetDataframe['date_time'])
        target_df = targetDataframe.set_index('date_time')
        samplingFrequency = getSamplingFreq(target_df)
        futureDates = pd.date_range(
            start=pd.Timestamp(target_df.index[-1]) + pd.Timedelta(samplingFrequency),
            periods=1,
            freq=samplingFrequency)
        return pd.DataFrame({'date_time': futureDates, 'pred': predictionValue})
    
    