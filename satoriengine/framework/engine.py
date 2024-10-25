import threading
import joblib
import os
from reactivex.subject import BehaviorSubject
from satorilib.api.time import datetimeToTimestamp, now
from satorilib.api.hash import generatePathId
from satorilib.concepts import Stream, StreamId, Observation
from satorilib.api.hash import hashIt
from satorilib.api.disk import getHashBefore

import pandas as pd
from datetime import datetime
import random
from typing import Union, Optional, List, Any, Dict
from collections import namedtuple

from satoriengine.framework.process import process_data
from satoriengine.framework.determine_features import determine_feature_set
from satoriengine.framework.model_creation import model_create_train_test_and_predict
from satoriengine.framework.structs import StreamForecast


class Engine:
    def __init__(self, streams: list[Stream], pubstreams: list[Stream]):
        self.streams = streams
        self.pubstreams = pubstreams
        self.streamModels: Dict[StreamId, StreamModel] = {}
        self.new_observation: BehaviorSubject = BehaviorSubject(None)
        self.prediction_produced: BehaviorSubject = BehaviorSubject(None)
        self.setup_subscriptions()
        self.initialize_models()

    def setup_subscriptions(self):
        self.new_observation.subscribe(
            on_next=lambda x: self.handle_new_observation(x) if x is not None else None,
            on_error=lambda e: self.handle_error(e),
            on_completed=lambda: self.handle_completion(),
        )

    def initialize_models(self):
        for stream in self.streams:
            self.streamModels[stream.streamId] = StreamModel(
                streamId=stream.streamId,
                prediction_produced=self.prediction_produced,
            )

    def handle_new_observation(self, observation: Observation):
        print(f"new_observation: {observation}")
        streamModel = self.streamModels.get(observation.streamId)
        if streamModel.thread is None or not streamModel.thread.is_alive():
            streamModel.choose_pipeline()
            streamModel.run_forever()
        if streamModel is not None:
            streamModel.produce_prediction()
            # streamModel.choose_pipeline()
        else:
            print(f"No model found for stream {observation.streamId}")

    def handle_error(self, error):
        print(f"An error occurred new_observaiton: {error}")

    def handle_completion(self):
        print(f"new_observation completed")


class StreamModel:
    def __init__(
        self,
        streamId: StreamId,
        prediction_produced: BehaviorSubject,
        datapath_override: str = None,
        modelpath_override: str = None,
    ):
        self.thread = None
        self.streamId = streamId
        self.datapath = datapath_override or self.data_path()
        self.modelpath = modelpath_override or self.model_path()
        self.pipeline: PipelineInterface = None
        self.stable: PipelineInterface = None
        self.prediction_produced = prediction_produced
        # self.last_backtest_error = None
        # self.stagnation_count = 0
        self.choose_pipeline(getStart=True) 

    def produce_prediction(self, updated_model=None):
        """
        triggered by
            - stable model replaced with a better one
            - new observation on the stream
        """
        updated_model = updated_model or self.stable
        if updated_model is not None:
            forecast = self.pipeline.predict(stable=self.stable, datapath=self.datapath)

            if isinstance(forecast, pd.DataFrame):
                observationTime = datetimeToTimestamp(now())
                prediction = StreamForecast.firstPredictionOf(forecast)
                observationHash = hashIt(
                    getHashBefore(pd.DataFrame(), observationTime)
                    + str(observationTime)
                    + str(prediction)
                )
                streamforecast = StreamForecast(
                    streamId=self.streamId,
                    forecast=forecast,
                    observationTime=observationTime,
                    observationHash=observationHash,
                )
                print("**************************")
                print(streamforecast)
                print("**************************")
                self.prediction_produced.on_next(streamforecast)

    def data_path(self) -> str:
        print(f"../../data/{generatePathId(streamId=self.streamId)}/aggregate.csv")
        return f"../../data/{generatePathId(streamId=self.streamId)}/aggregate.csv"

    def model_path(self) -> str:
        return f"../../models/{generatePathId(streamId=self.streamId)}"

    def load(self) -> Union[None, list]:
        """loads the stable model from disk if present"""
        try:
            self.stable = joblib.load(self.modelpath)
            return self.stable
        except FileNotFoundError:
            return None

    # def check_stagnation(self, pilot_model: Optional[Any]) -> bool:
    #     """
    #     Check if the backtest error has remained the same for multiple iterations
    #     Returns True if training should stop due to stagnation
    #     """
    #     current_error = pilot_model[0].backtest_error
    #     if self.last_backtest_error is None:
    #         self.last_backtest_error = current_error
    #         return False

    #     if abs(current_error - self.last_backtest_error) < 1e-10:
    #         self.stagnation_count += 1
    #     else:
    #         self.stagnation_count = 0

    #     self.last_backtest_error = current_error
    #     return self.stagnation_count >= 5

    def check_observations(self, has_header: bool = False) -> bool:
        """
        Check if a CSV file has fewer than 3 observations.
        
        Args:
            file_path (str): Path to the CSV file
            has_header (bool): Whether the csv contains a header
        Returns:
            bool: True if file has more than 2 rows, False otherwise
        """
        return len(pd.read_csv(self.datapath, nrows=3, header=0 if has_header else None)) > 2

    def choose_pipeline(self, getStart: bool = False):
        """
        everything can try to handle some cases
        Engine
            - low resources available - SKPipeline
            - few observations - SKPipeline
            - (mapping of cases to suitable pipelines)
        examples: StartPipeline, SKPipeline, XGBoostPipeline, ChronosPipeline, DNNPipeline
        refactor to build a simple StartPipeline
        called when the context may have changed
        startpipeline should check for it's own stagnation and return a flag on the trainingResult object
        """
        if getStart:
            self.pipeline = None
        else:
            if self.check_observations(): 
                self.pipeline = SKPipeline
            else:
                self.pipeline = StarterPipeline

    def run(self):
        """
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        Breaks if backtest error stagnates for 3 iterations.
        """
        while True:
            trainingResult = self.pipeline.train(
                stable=self.stable, 
                datapath=self.datapath
            )
            if trainingResult.status == 1 and not trainingResult.stagnated:
                if self.pipeline.compare(self.stable, trainingResult.model):
                    if self.pipeline.save(trainingResult.model, self.modelpath):
                        self.stable = trainingResult.model
                        self.produce_prediction(self.stable)
            else:
                break

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()


class TrainingResult:

    def __init__(
        self, status, model: "PipelineInterface", stagnated: bool = False
    ):
        self.status = status
        self.model = model
        self.stagnated = stagnated


class PipelineInterface:
    @staticmethod
    def compare(
        stable: Optional[Any], comparison: bool = False, replace: bool = False
    ) -> bool:
        """
        Compare stable and pilot models based on their backtest error.

        Args:
            stable: The current stable model
            replace: Whether to replace stable with pilot if pilot performs better

        Returns:
            bool: True if pilot should replace stable, False otherwise
        """
        pass

    @staticmethod
    def predict(**kwargs) -> Union[None, pd.DataFrame]:
        """
        Make predictions using the stable model

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            Optional[pd.DataFrame]: Predictions if successful, None otherwise
        """
        pass

    @staticmethod
    def save(model: Optional[Any], modelpath: str) -> bool:
        """
        Save the model to disk.

        Args:
            model: The model to save
            modelpath: Path where the model should be saved

        Returns:
            bool: True if save successful, False otherwise
        """
        pass

    @staticmethod
    def train(**kwargs) -> TrainingResult:
        """
        Train a new model.

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            TrainingResult: Object containing training status and model
        """
        pass


class SKPipeline(PipelineInterface):
    @staticmethod
    def train(**kwargs) -> TrainingResult:
        if kwargs["stable"] is None:
            status, model = SKPipeline.skEnginePipeline(
                kwargs["datapath"], ["quick_start"]
            )
            if status == 1:
                return TrainingResult(status, model, False)
        status, model = SKPipeline.skEnginePipeline(kwargs["datapath"], ["random_model"])
        return TrainingResult(status, model, False)

    @staticmethod
    def save(model: Optional[Any], modelpath: str) -> bool:
        """saves the stable model to disk"""
        try:
            os.makedirs(os.path.dirname(modelpath), exist_ok=True)
            joblib.dump(model, modelpath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

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
        status, predictor_model = SKPipeline.skEnginePipeline(
            filename=kwargs["datapath"],
            list_of_models=[kwargs["stable"][0].model_name],
            mode="predict",
            unfitted_forecaster=kwargs["stable"][0].unfitted_forecaster,
        )
        if status == 1:
            return predictor_model[0].forecast
        return None

    @staticmethod
    def skEnginePipeline(
        filename: str,
        list_of_models: List[str],
        interval: List[int] = [10, 90],
        feature_set_reduction: bool = False,
        exogenous_feature_type: str = "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
        feature_set_reduction_method: str = "RFECV",
        random_state_hyperr: int = 123,
        metric: str = "mase",
        mode: str = "train",
        unfitted_forecaster: Optional[Any] = None,
    ):
        """Engine function for the Satori Engine"""

        def check_model_suitability(list_of_models, allowed_models, dataset_length):
            suitable_models = []
            unsuitable_models = []
            for model in list_of_models:
                if model in allowed_models:
                    suitable_models.append(True)
                else:
                    suitable_models.append(False)
                    reason = f"Not allowed for dataset size of {dataset_length}"
                    unsuitable_models.append((model, reason))
            return suitable_models, unsuitable_models

        list_of_models = [model.lower() for model in list_of_models]

        quick_start_present = "quick_start" in list_of_models
        random_model_present = "random_model" in list_of_models
        random_state_hyper = random_state_hyperr

        # Process data first to get allowed_models
        proc_data = process_data(filename, quick_start=quick_start_present)

        # if quick_start_present and random_model_present:
        #     warnings.warn(
        #         "Both 'quick_start' and 'random_model' are present. 'quick_start' will take precedence.")

        if random_model_present and not quick_start_present:
            current_time = datetime.now()
            seed = int(current_time.strftime("%Y%m%d%H%M%S%f"))
            random.seed(seed)
            print(f"Using random seed: {seed}")

            # Randomly select options
            feature_set_reduction = random.choice([True, False])
            feature_set_reduction = False
            exogenous_feature_type = random.choice(
                [
                    "NoExogenousFeatures",
                    "Additive",
                    "AdditiveandMultiplicativeExogenousFeatures",
                    "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
                ]
            )
            feature_set_reduction_method = random.choice(["RFECV", "RFE"])
            random_state_hyper = random.randint(0, 2**32 - 1)

            # Replace 'random_model' with a randomly selected model from allowed_models
            list_of_models = [
                (
                    random.choice(proc_data.allowed_models)
                    if model == "random_model"
                    else model
                )
                for model in list_of_models
            ]
            print(f"Randomly selected models: {list_of_models}")
            print(f"feature_set_reduction: {feature_set_reduction}")
            print(f"exogenous_feature_type: {exogenous_feature_type}")
            print(f"feature_set_reduction_method: {feature_set_reduction_method}")
            print(f"random_state_hyper: {random_state_hyper}")

        if quick_start_present:
            feature_set_reduction = False
            exogenous_feature_type = "NoExogenousFeatures"
            list_of_models = proc_data.allowed_models

        if proc_data.if_invalid_dataset:
            return 2, "Status = 2 (insufficient amount of data)"

        # Check if the requested models are suitable based on the allowed_models
        suitable_models, unsuitable_models = check_model_suitability(
            list_of_models, proc_data.allowed_models, len(proc_data.dataset)
        )

        if unsuitable_models:
            print("The following models are not allowed due to insufficient data:")
            for model, reason in unsuitable_models:
                print(f"- {model}: {reason}")

        if not any(suitable_models):
            return (
                3,
                "Status = 3 (none of the requested models are suitable for the available data)",
            )

        # Filter the list_of_models to include only suitable models
        list_of_models = [
            model
            for model, is_suitable in zip(list_of_models, suitable_models)
            if is_suitable
        ]

        try:
            features = None
            for model_name in list_of_models:
                if model_name in ["baseline", "arima"]:
                    features = determine_feature_set(
                        dataset=proc_data.dataset,
                        data_train=proc_data.data_subsets["train"],
                        end_validation=proc_data.end_times["validation"],
                        end_train=proc_data.end_times["train"],
                        dataset_with_features=proc_data.dataset_withfeatures,
                        dataset_start_time=proc_data.dataset_start_time,
                        dataset_end_time=proc_data.dataset_end_time,
                        initial_lags=proc_data.lags,
                        weight_para=proc_data.use_weight,
                        exogenous_feature_type=exogenous_feature_type,
                        feature_set_reduction=feature_set_reduction,
                        feature_set_reduction_method=feature_set_reduction_method,
                        bayesian_trial=20,
                        random_state_hyper=random_state_hyper,
                        frequency=proc_data.sampling_frequency,
                        backtest_steps=proc_data.backtest_steps,
                        prediction_steps=proc_data.forecasting_steps,
                        hyper_flag=False,
                    )
                else:
                    features = determine_feature_set(
                        dataset=proc_data.dataset,
                        data_train=proc_data.data_subsets["train"],
                        end_validation=proc_data.end_times["validation"],
                        end_train=proc_data.end_times["train"],
                        dataset_with_features=proc_data.dataset_withfeatures,
                        dataset_start_time=proc_data.dataset_start_time,
                        dataset_end_time=proc_data.dataset_end_time,
                        initial_lags=proc_data.lags,
                        weight_para=proc_data.use_weight,
                        exogenous_feature_type=exogenous_feature_type,
                        feature_set_reduction=feature_set_reduction,
                        feature_set_reduction_method=feature_set_reduction_method,
                        bayesian_trial=20,
                        random_state_hyper=random_state_hyper,
                        frequency=proc_data.sampling_frequency,
                        backtest_steps=proc_data.backtest_steps,
                        prediction_steps=proc_data.forecasting_steps,
                        hyper_flag=True,
                    )

            list_of_results = []
            for model_name in list_of_models:
                result = model_create_train_test_and_predict(
                    model_name=model_name,
                    dataset=proc_data.dataset,
                    dataset_train=proc_data.data_subsets["train"],
                    end_validation=proc_data.end_times["validation"],
                    end_test=proc_data.end_times["test"],
                    sampling_freq=proc_data.sampling_frequency,
                    differentiation=features.differentiation,
                    selected_lags=features.selected_lags,
                    selected_exog=features.selected_exog,
                    dataset_selected_features=features.dataset_selected_features,
                    data_missing=features.missing_values,
                    weight=features.weight,
                    select_hyperparameters=True,
                    default_hyperparameters=None,
                    random_state_hyper=random_state_hyper,
                    backtest_steps=proc_data.backtest_steps,
                    interval=interval,
                    metric=metric,
                    forecast_calendar_features=features.forecast_calendar_features,
                    forecasting_steps=proc_data.forecasting_steps,
                    hour_seasonality=features.hour_seasonality,
                    dayofweek_seasonality=features.dow_seasonality,
                    week_seasonality=features.week_seasonality,
                    baseline_1=proc_data.time_metric_baseline,
                    baseline_2=proc_data.forecasterequivalentdate,
                    baseline_3=proc_data.forecasterequivalentdate_n_offsets,
                    mode=mode,
                    forecaster=unfitted_forecaster,
                )
                list_of_results.append(result)

            return 1, list_of_results  # Status = 1 (ran correctly)

        except Exception as e:
            # Additional status code for unexpected errors
            return 4, f"An error occurred: {str(e)}"


class StarterPipeline(PipelineInterface):
    @staticmethod
    def train(**kwargs) -> TrainingResult:
        if kwargs["stable"] is None:
            status, model = StarterPipeline.starterEnginePipeline(
                kwargs["datapath"]
            )
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
        status, predictor_model = StarterPipeline.starterEnginePipeline(
            filename=kwargs["datapath"]
        )
        if status == 1:
            return predictor_model[0].forecast
        return None

    @staticmethod
    def starterEnginePipeline(
        filename: str
    ):
        """Starter Engine function for the Satori Engine"""

        col_names_starter = ["date_time", "value", "id"]
        starter_dataset = pd.read_csv(filename, names=col_names_starter, header=None)
        
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
