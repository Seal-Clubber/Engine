import pandas as pd
import numpy as np
from typing import Union
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from satorilib.logging import info, debug
from Engine.satoriengine.veda.adapters.multivariate.data import conformData, createTrainTest, getSamplingFreq


class LightMVAdapter(ModelAdapter):

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
        self.model: Union[LightMVAdapter, None] = None
        self.dataTrain: pd.DataFrame = pd.DataFrame()
        self.fullDataset: pd.DataFrame = pd.DataFrame()
        self.modelError: float = 0
        self.covariateColNames: list[str] = []
        self.forecastingSteps: int = 1
        self.rng = np.random.default_rng(37)

    def load(self, modelPath: str, **kwargs) -> Union[None, "ModelAdapter"]:
        """loads the model model from disk if present"""
        pass

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        pass

    def fit(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> TrainingResult:
        self._manageData(targetData, covariateData)
        self.model = self._multivariateFit()
        self.model.refit_full(model = 'best', set_best_to_refit_full = True)
        return TrainingResult(1, self)
    
    def compare(self, other: ModelAdapter, **kwargs) -> bool:
        return True

    def score(self, **kwargs) -> float:
        return 0.0

    def predict(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame], **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        self._manageData(targetData, covariateData)
        datsetWithFutureCov = self.appendCovariateFuture(self.fullDataset)
        prediction = self.model.predict(self.fullDataset, known_covariates=datsetWithFutureCov.drop('value', axis=1))
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
            # random_seed=self.rng, 
            hyperparameters={
            "Naive": {}, 
            "SeasonalNaive": {},
            "Average": {}, 
            "SeasonalAverage": {},
            "NPTS": {}, 
            "DirectTabular": {
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "WithLighGBMRegressor"}, 
            },
            "RecursiveTabular": {
                    "target_scaler": "standard", 
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "WithLighGBMRegressor"}, 
            },
            "Chronos": [
                {
                    "model_path": "bolt_base", 
                    "ag_args": {"name_suffix": "ZeroShot"}
                }, 
                {
                    "model_path": "bolt_base", 
                    "covariate_regressor": "XGB",
                    "target_scaler": "standard", 
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "WithXGBRegressor"}, 
                },
            ],
            "SimpleFeedForward": { 
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
            }, 
            "DLinear": { 
                    "target_scaler": "standard", 
                    "covariate_scaler": "global",
            }, 
            "PatchTST": { 
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
            }, 
            "DeepAR": { 
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
            }, 
            "TemporalFusionTransformer": [ 
                { 
                    "disable_static_features": False,
                    "disable_known_covariates": False, 
                    "disable_past_covariates": False,
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "UseAllCovariates"}, 
                },
                {
                    "disable_static_features": False,
                    "disable_known_covariates": True, 
                    "disable_past_covariates": False,
                    "target_scaler": "standard", 
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "UseOnlyPastCovariates"}, 
                },
                {  
                    "disable_static_features": False,
                    "disable_known_covariates": False, 
                    "disable_past_covariates": True,
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "UseOnlyKnownCovariates"}, 
                },
                { 
                    "disable_static_features": True,
                    "disable_known_covariates": True, 
                    "disable_past_covariates": True,
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "IgnoreAllCovariates"}, 
                },
            ],
            # "TiDE": [ 
            #     { 
            #         "disable_static_features": False,
            #         "disable_known_covariates": False, 
            #         "disable_past_covariates": False,
            #         "target_scaler": "standard",
            #         "covariate_scaler": "global",
            #         "ag_args": {"name_suffix": "UseKnownCovariates"}, 
            #     },
            # ],
            "WaveNet": [
                {
                    "disable_static_features": False,
                    "disable_known_covariates": False, 
                    "disable_past_covariates": False,
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "UseKnownCovariates"}, 
                },
                { 
                    "disable_static_features": False,
                    "disable_known_covariates": True, 
                    "disable_past_covariates": True,
                    "target_scaler": "standard",
                    "covariate_scaler": "global",
                    "ag_args": {"name_suffix": "IgnoreAllCovariates"}, 
                },
            ], 
        },
            num_val_windows = 1, 
            val_step_size = 7, 
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
