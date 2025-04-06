import pandas as pd
import numpy as np
from typing import Union
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from satorilib.logging import info, debug
from satoriengine.veda.adapters.multivariate.data import conformData, createTrainTest, getSamplingFreq


class HeavyMVAdapter(ModelAdapter):

    @staticmethod
    def condition(*args, **kwargs) -> float:
        # TODO: change this appropriately
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
        self.model: Union[HeavyMVAdapter, None] = None
        self.dataTrain: pd.DataFrame = pd.DataFrame()
        self.fullDataset: pd.DataFrame = pd.DataFrame()
        self.modelError: float = 0.0
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
        impFeatures = self.model.feature_importance(self.fullDataset, relative_scores=True)
        maxImportance = impFeatures['importance'].max()
        # Here we update the the original co-variate column list with the important feature list
        self.covariateColNames = maxImportance[impFeatures['importance'] > maxImportance * 0.01].index.tolist()
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
        datsetWithFutureCov = self.appendCovariateFuture(self.fullDataset, self.covariateColNames)
        prediction = self.model.predict(self.fullDataset, known_covariates=datsetWithFutureCov.tail(1).drop('value', axis=1))
        resultDf = self._getPredictionDataframe(targetData, prediction.mean()[0]) # TODO: can also use in-built auto-gluon stuff ( optimize )
        return resultDf
    
    def _manageData(self, targetData: pd.DataFrame, covariateData: list[pd.DataFrame]):
        conformedData, self.covariateColNames = conformData(targetData, covariateData) 
        self.dataTrain, self.fullDataset = createTrainTest(conformedData, self.forecastingSteps)

    def _multivariateFit(self) -> TimeSeriesPredictor:
        return TimeSeriesPredictor(
            prediction_length=self.forecastingSteps, # if we change the forecasting steps, the other variables will have to change accordingly
            eval_metric="MASE",
            target="value", 
            known_covariates_names=self.covariateColNames,
            verbosity=0
        ).fit(
            self.fullDataset,
            # random_seed=self.rng, 
            hyperparameters={
            # hyperparameter settings for better Satori adapter
            # Baseline models
            "Naive": {}, # Name: "Naive" , univariate baseline model, last value
            "SeasonalNaive": {}, # Name: "SeasonalNaive", univariate baseline model, last value in comparable cyclical season
            "Average": {}, # Name: "Average", univariate average baseline model
            "SeasonalAverage": {}, # Name: "SeasonalAverage", univariate seasonal average baseline model
            "NPTS": {}, # Name: "NPTS", univariate probabilistic baseline forecaster
            # # Local models (based on classical time series statistics)
            # # All local models perform forecasting slower than other types of models and were not suitable for the VPS adapter.
            # # Redundant with AutoETS
            # "ETS": [ 
            #     {}, # Name: "ETS", univariate default version of ETS
            #     # Name: "ETSWithXGBRegressor", A hybrid ETS model combined with an XGB model that
            #     # accepts related exogeous feature data to predict the target value
            #     {
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
            #         "ag_args": {"name_suffix": "WithXGBRegressor"},
            #     },
            # ],
            "AutoETS": [ 
                {}, # Name: "AutoETS", univariate ETS with automatic hyperparameter selection
            #     # Name: "AutoETSWithXGBRegressor"
            #     # Limited value add for the hybrid
            #     # A hybrid ETS model (automatical hyperparameter selection) combined with an XGB model that
            #     # accepts related exogeous feature data to predict the target value
            #     {
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parameter must be set on a model by model level
            #         "ag_args": {"name_suffix": "WithXGBRegressor"},
            #     },
            ],
            # # Redundant with AutoARIMA
            # "ARIMA": [
            #     {}, # Name: "ARIMA", univariate ARIMA
            #     # Name: ARIMAWithXGBRegressor"
            #     # A hybrid univariate ARIMA model combined with an XGB model that
            #     # accepts related exogeous feature data to predict the target value
            #     {
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
            #         "ag_args": {"name_suffix": "WithXGBRegressor"},
            #     },
            # ],
            "AutoARIMA": [
                {}, # Name: "AutoARIMA", univariate ARIMA with automatic hyperparameter selection
            #     # Name: "AutoARIMAWithXGBRegressor"
            #     # Limited value add for the hybrid
            #     # A hybrid univariate ARIMA model (automatic hyperparameter selection) combined with 
            #     # an XGB model that accepts related exogeous feature data to predict the target value
            #     {
            #         "covariate_regressor": "XGB",# Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parameter must be set on a model by model level
            #         "ag_args": {"name_suffix": "WithXGBRegressor"},
            #     },
            ],
            "AutoCES": [
                {}, # Name: "AutoCES", univariate CES with automatic hyperparameter selection
            #     # Name: "AutoCESWithXGBRegressor"
            #     # Limited value add for the hybrid
            #     # A hybrid univariate AutoCES model combined with an XGB model that
            #     # accepts related exogeous feature data to predict the target value
            #     {
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parameter must be set on a model by model level
            #         "ag_args": {"name_suffix": "WithXGBRegressor"},
            #     },
            ],
            # # Redundant with DynamicOptimizedTheta
            # "Theta": [
            #     # Name: "ThetaMultiplicative"
            #     # univariate Theta with multiplicative decomposition
            #     {
            #         "decomposition_type": "multiplicative",
            #         "target_scaler": "standard", # Scale targets, This parameter must be set on a model by model level
            #         "ag_args": {"name_suffix": "Multiplicative"},
            #     },
            #     # Name: "ThetaAdditive" 
            #     # univariate Theta with additive decomposition
            #     { 
            #         "decomposition_type": "additive",
            #         "target_scaler": "standard", # Scale targets, This parameter must be set on a model by model level
            #         "ag_args": {"name_suffix": "Additive"},
            #     },
            #     # Name: "ThetaMultiplicativeWithXGBRegressor"
            #     # A hybrid univariate Theta model with multiplicative decomposition, 
            #     # combined with an XGB model that accepts related exogeous feature data to predict the target value
            #     {  
            #         "decomposition_type": "multiplicative",
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
            #         "ag_args": {"name_suffix": "MultiplicativeWithXGBRegressor"},
            #     },
            #     # Name: "ThetaAdditiveWithXGBRegressor" 
            #     # A hybrid univariate Theta model with additive decomposition, 
            #     # combined with an XGB model that accepts related exogeous feature data to predict the target value
            #     {
            #         "decomposition_type": "additive",
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
            #         "ag_args": {"name_suffix": "AdditiveWithXGBRegressor"},
            #     },
            # ],
            "DynamicOptimizedTheta": [
                # Name: "DynamicOptimizedThetaMultiplicative"
                # univariate optimized version of Theta with multiplicative decomposition
                {
                    "decomposition_type": "multiplicative",
                    "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
                    "ag_args": {"name_suffix": "Multiplicative"},
                },
                # Name: "DynamicOptimizedThetaAdditive"
                # univariate optimized version of  Theta with additive decomposition
                { 
                    "decomposition_type": "additive",
                    "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
                    "ag_args": {"name_suffix": "Additive"},
                },
            #     # Name: "DynamicOptimizedThetaMultiplicativeWithXGBRegressor"
            #     # Limited value add for the hybrid
            #     # A hybrid univariate optimized version of Theta model with multiplicative decomposition, 
            #     # combined with an XGB model that accepts related exogeous feature data to predict the target value
            #     {  
            #         "decomposition_type": "multiplicative",
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
            #         "ag_args": {"name_suffix": "MultiplicativeWithXGBRegressor"},
            #     },
            #     # Name: "DynamicOptimizedThetaAdditiveWithXGBRegressor"
            #     # Limited value add for the hybrid
            #     # A hybrid univariate optimized version of Theta model with additive decomposition, 
            #     # combined with an XGB model that accepts related exogeous feature data to predict the target value
            #     {
            #         "decomposition_type": "additive",
            #         "covariate_regressor": "XGB", # Use XGB regressor for the hybrid model
            #         "target_scaler": "standard", # Scale targets, This parametger must be set on a model by model level
            #         "ag_args": {"name_suffix": "AdditiveWithXGBRegressor"},
            #     },
            ], 
            # Tabular models
            # Name: "DirectTabularWithLighGBMRegressor"
            # A direct tabular model that uses the default of lightgbm and can accept covriate data
            "DirectTabular": {                        
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training process
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "WithLightGBMRegressor"}, # Custom name
            },
            # Name: "RecursiveTabularWithLightGBMRegressor" 
            # A recursive tabular model that uses the default of lightgbm and can accept covriate data
            "RecursiveTabular": {
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "WithLightGBMRegressor"}, # Custom name
            },
            "Chronos": [
                # Many of the Chronos bolt models produce similar results
                # The VPS and better adapters includes two of the models: bolt_base and the bolt_base with covariates
                # Depending on the data set, one could also use the fine tune trained versions of bolt_small and bolt_small
                # with covariates. However, the fine tune trained versions of bolt_base and bolt_base with covariates are
                # extremely memory and disk space intensive and not recommended for most use. If suitable, they could be
                # fine tune trained off-line and then uploaded for use.
                # # Name: "ChronosZeroShot[bolt_tiny]"
                # # A Zero Shot version of Chonos bolt_tiny
                # {"model_path": "bolt_tiny", "ag_args": {"name_suffix": "ZeroShot"}},
                # # Name: "ChronosFineTune[bolt_tiny]"
                # # A fine tune trained version of Chonos bolt_tiny
                # {"model_path": "bolt_tiny", "fine_tune": True, "ag_args": {"name_suffix": "FineTune"}},
                # # Name: "ChronosWithXGBRegressor[bolt_tiny]"
                # # Model: A hybrid Chronos bolt_tiny model combined with an XGB model that
                # # accepts covariate data to predict the target value
                # {
                #     "model_path": "bolt_tiny", # which bolt model
                #     "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                #     "target_scaler": "standard", # scales the target values
                #     "covariate_scaler": "global", # scales the covariates
                #     "ag_args": {"name_suffix": "WithXGBRegressor"}, # Custom name
                # },
                # # Name: "ChronosFineTuneWithXGBRegressor[bolt_tiny]"
                # # Model: A hybrid Chronos bolt_tiny model combined with an XGB model that
                # # accepts covariate data to predict the target value
                # # Includes fine tune training
                # {
                #     "model_path": "bolt_tiny", # which bolt model
                #     "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                #     "target_scaler": "standard", # scales the target values
                #     "covariate_scaler": "global", # scales the covariates
                #     "fine_tune": True, # set to fine tune train the model
                #     "ag_args": {"name_suffix": "FineTuneWithXGBRegressor"}, # Custom name
                # },
                # # Name: "ChronosZeroShot[bolt_mini]"
                # # A Zero Shot version of Chonos bolt_mini
                # {"model_path": "bolt_mini", "ag_args": {"name_suffix": "ZeroShot"}},
                # # Name: "ChronosFineTune[bolt_mini]"
                # # A fine tune trained version of Chonos bolt_mini
                # {"model_path": "bolt_mini", "fine_tune": True, "ag_args": {"name_suffix": "FineTune"}},
                # # Name: "ChronosWithXGBRegressor[bolt_mini]"
                # # Model: A hybrid Chronos bolt_mini model combined with an XGB model that
                # # accepts covariate data to predict the target value
                # {
                #     "model_path": "bolt_mini", # which bolt model
                #     "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                #     "target_scaler": "standard", # scales the target values
                #     "covariate_scaler": "global", # scales the covariates
                #     "ag_args": {"name_suffix": "WithXGBRegressor"}, # Custom name
                # },
                # # Name: "ChronosFineTuneWithXGBRegressor[bolt_mini]"
                # # Model: A hybrid Chronos bolt_mini model combined with an XGB model that
                # # accepts covariate data to predict the target value
                # # Includes fine tune training
                # {
                #     "model_path": "bolt_mini", # which bolt model
                #     "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                #     "target_scaler": "standard", # scales the target values
                #     "covariate_scaler": "global", # scales the covariates
                #     "fine_tune": True, # set to fine tune train the model
                #     "ag_args": {"name_suffix": "FineTuneWithXGBRegressor"}, # Custom name
                # },
                # # Name: "ChronosZeroShot[bolt_small]"
                # # A Zero Shot version of Chonos bolt_small
                # {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
                # # Name: "ChronosFineTune[bolt_small]
                # # A fine tune trained version of Chonos bolt_small
                # {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTune"}},
                # # Name: "ChronosWithXGBRegressor[bolt_small]"
                # # Model: A hybrid Chronos bolt_small model combined with an XGB model that
                # # accepts covariate data to predict the target value
                # {
                #     "model_path": "bolt_small", # which bolt model
                #     "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                #     "target_scaler": "standard", # scales the target values
                #     "covariate_scaler": "global", # scales the covariates
                #     "ag_args": {"name_suffix": "WithXGBRegressor"}, # Custom name
                # },
                # # Name: "ChronosFineTuneWithXGBRegressor[bolt_small]"
                # # Model: A hybrid Chronos bolt_small model combined with an XGB model that
                # # accepts covariate data to predict the target value
                # # Includes fine tune training
                # {
                #     "model_path": "bolt_small", # which bolt model
                #     "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                #     "target_scaler": "standard", # scales the target values
                #     "covariate_scaler": "global", # scales the covariates
                #     "fine_tune": True, # set to fine tune train the model
                #     "ag_args": {"name_suffix": "FineTuneWithXGBRegressor"}, # Custom name
                # },
                # Name: "ChronosZeroShot[bolt_base]"
                # A Zero Shot version of Chonos bolt_base
                {"model_path": "bolt_base", "ag_args": {"name_suffix": "ZeroShot"}},
                # # Name: "ChronosFineTune[bolt_base]"
                # # A fine tune trained version of Chonos bolt_base
                # {"model_path": "bolt_base", "fine_tune": True, "ag_args": {"name_suffix": "FineTune"}},
                # Name: "ChronosWithXGBRegressor[bolt_base]"
                # Model: A hybrid Chronos bolt_base model combined with an XGB model that
                # accepts covariate data to predict the target value
                {
                    "model_path": "bolt_base", # which bolt model
                    "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                    "target_scaler": "standard", # scales the target values
                    "covariate_scaler": "global", # scales the covariates
                    "ag_args": {"name_suffix": "WithXGBRegressor"}, # Custom name
                },
                # # Name: "ChronosFineTuneWithXGBRegressor[bolt_base]"
                # # Model: A hybrid Chronos bolt_base model combined with an XGB model that
                # # accepts covariate data to predict the target value
                # # Fine tuned
                # {
                #     "model_path": "bolt_base", # which bolt model
                #     "covariate_regressor": "XGB", # regressor used to incorporate covariates into hybrid model
                #     "target_scaler": "standard", # # scales the target values
                #     "covariate_scaler": "global", # scales the covariates
                #     "fine_tune": True, # set to fine tune train the model
                #     "ag_args": {"name_suffix": "FineTuneWithXGBRegressor"}, # Custom name
                # },
            ],
            # Deep learning models that are trained (not LLM)
            # Name: "SimpleFeedForward"
            # A simple feedforward artificial neural network that uses covariates
            "SimpleFeedForward": { 
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
            }, 
            # Name: "DLinear"
            # simple feedforward artificial neural network, which firsts subtracts trend, and also uses covariates
            "DLinear": { 
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
            }, 
            # Name: "PatchTST"
            # An artificial neural network model that divides time series into regions and uses covariates
            "PatchTST": { 
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
            }, 
            # Google's highly specialized deep learning architecture that allows the model
            # to use or not use different types of input data
            "TemporalFusionTransformer": [ 
                # Name: "TemporalFusionTransformerUseAllCovariates"
                # This configuration of the model uses future known covariates but does not use past covariates
                # Technically, regardless of the name that I chose, this can also use static features (not in current data)
                {  
                    "disable_static_features": False,
                    "disable_known_covariates": False, 
                    "disable_past_covariates": False,
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "UseAllCovariates"}, # Custom name
                },
                # Name: "TemporalFusionTransformerUseOnlyPastCovariates"
                # This configuration of the model uses all available covariates
                { 
                    "disable_static_features": False,
                    "disable_known_covariates": True, 
                    "disable_past_covariates": False,
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "UseOnlyPastCovariates"}, # Custom name
                },
                # Name: "TemporalFusionTransformerUseOnlyKnownCovariates"
                # This configuration of the model uses past covariates but does not use future known covariates
                # Technically, regardless of the name that I chose, this can also use static features (not in current data)
                { 
                    "disable_static_features": False,
                    "disable_known_covariates": False, 
                    "disable_past_covariates": True,
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "UseOnlyKnownCovariates"}, # Custom name
                },
                # Name: "TemporalFusionTransformerIgnoreAllCovariates"
                # This configuration of the model does not use any covariates, even if available
                {
                    "disable_static_features": True,
                    "disable_known_covariates": True, 
                    "disable_past_covariates": True,
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "IgnoreAllCovariates"}, # Custom name
                },
            ],
            # DeepAR model
            # The current Autogluon implementation of the DeepAR model may cause erratic behavior in the scheduler and
            # may be turned off
            # Name: "DeepAR"
            # DeepAR model that uses known covariates
            # "DeepAR": {},
            # Time series Dense Encoder (TiDE) model    
            # The current Autogluon implementation of the TiDE model may cause erratic behavior in the scheduler.
            # Therefore, it is not turned on, unless the model runs for a long time.
            # "TiDE": [ 
            #     # Name: "TiDEUseKnownCovariates"
            #     # This configuration of the model uses known future covariates (implementation does not support past ones)
            #    {
            #        "disable_static_features": False,
            #        "disable_known_covariates": False, 
            #        "disable_past_covariates": False,
            #        "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
            #        "covariate_scaler": "global", # Scale the covariates
            #        "ag_args": {"name_suffix": "UseKnownCovariates"}, # Custom name
            #    },
            #     # Name: "TiDEIgnoreCovariates"
            #     # This configuration of the model ignores known future covariates (implementation does not support past ones)
            #    {
            #        "disable_static_features": True,
            #        "disable_known_covariates": True, 
            #        "disable_past_covariates": True,
            #        "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
            #        "covariate_scaler": "global", # Scale the covariates
            #        "ag_args": {"name_suffix": "IgnoreCovariates"}, # Custom name
            #    },
            # ],
            # A specialized Convolutional Neural Network
            "WaveNet": [ 
                # Name: "WaveNetIgnoreAllCovariates"
                # This configuration of the model ignores covariates
                { 
                    "disable_static_features": True,
                    "disable_known_covariates": True, 
                    "disable_past_covariates": True,
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "IgnoreAllCovariates"}, # Custom name
                },
                # Name: "WaveNetUseKnownCovariates"
                # This configuration of the model uses known future covariates (implementation does not support past ones)
                { 
                    "disable_static_features": False,
                    "disable_known_covariates": False, 
                    "disable_past_covariates": False,
                    "target_scaler": "standard", # Scale the value we are trying to predict during the training proces
                    "covariate_scaler": "global", # Scale the covariates
                    "ag_args": {"name_suffix": "UseKnownCovariates"}, # Custom name
                },
            ], 
            # # Name: "ADIDA", Aggregate-Disaggregate Intermittent Demand Approach (ADIDA)
            # # removes intermittence by aggregating a time series into buckets that contain totals for different time ranges
            # "ADIDA": {}, 
            # # Name: "Croston",
            # # Calculates demand rates for periods and estimates periods where there will be no demand
            # "Croston": {}, 
            # # Name: "IMAPA", Intermittent Multiple Aggregation Prediction Algorithm (IMAPA) model
            # # aggregates the time series values at regular intervals to enable forecasting of the aggregated values.
            # "IMAPA": {}, 
        },
            num_val_windows = 7, # no.of backtests
            val_step_size = None, # step size bw backtests
            refit_every_n_windows = None,
            time_limit=3600, 
            enable_ensemble=True,
        )
    
    @staticmethod
    def appendCovariateFuture(df: pd.DataFrame, covariateColNameList: list[str]) -> pd.DataFrame:
        last_timestamp = df.index.get_level_values('timestamp').max()
        time_diff = df.index.get_level_values('timestamp').to_series().diff().mode()[0]
        next_timestamp = pd.Timestamp(last_timestamp) + time_diff
        item_id = df.index.get_level_values('item_id')[0]
        new_row_data = {}
        for colName in covariateColNameList:
            covDf = df[[colName]]
            if not covDf.empty:
                try:
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
                    new_row_data[colName] = prediction.mean()[0]
                except Exception as e:
                    if not covDf.empty:
                        new_row_data[colName] = covDf[colName].iloc[-1]
        new_row = pd.DataFrame(index=[(item_id, next_timestamp)], 
                            columns=df.columns)
        for col, value in new_row_data.items():
            new_row[col] = value
        if not isinstance(new_row.index, pd.MultiIndex):
            new_row.index = pd.MultiIndex.from_tuples([(item_id, next_timestamp)], 
                                                    names=df.index.names)
        formatted_index = pd.MultiIndex.from_tuples([(item_id, next_timestamp)], 
                                                names=df.index.names)
        new_row.index = formatted_index
        result_df = pd.concat([df, new_row])
        try:
            if isinstance(df, TimeSeriesDataFrame):
                result_df = TimeSeriesDataFrame(result_df)
        except (ImportError, Exception) as e:
            pass
        return result_df

    
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
