from typing import Union, Dict
import copy
import json
import threading
import pandas as pd
from reactivex.subject import BehaviorSubject
from satorilib.api.hash import hashIt
from satorilib.api.disk import getHashBefore
from satorilib.api.hash import generatePathId
from satorilib.api.time import datetimeToTimestamp, now
from satorilib.concepts import Stream, StreamId, Observation
from satoriengine.framework.structs import StreamForecast
from satoriengine.framework.pipelines.interface import PipelineInterface
from satoriengine.framework.pipelines.sk import SKPipeline
from satoriengine.framework.pipelines.starter import StarterPipeline


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
        # print(f"new_observation: {observation}")
        streamModel = self.streamModels.get(observation.streamId)
        streamModel.handle_new_observation(observation)
        if streamModel.thread is None or not streamModel.thread.is_alive():
            streamModel.choose_pipeline(inplace=True) # also should change the pilot model pipeline
            streamModel.run_forever()
        if streamModel is not None:
            streamModel.produce_prediction()
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
    ):
        self.thread = None
        self.streamId = streamId
        self.prediction_produced = prediction_produced
        self.data: pd.DataFrame = self.load_data()
        self.pipeline: PipelineInterface = self.choose_pipeline()
        self.pilot: PipelineInterface = self.pipeline.load(
            self.model_path()
        )
        self.stable: PipelineInterface = copy.deepcopy(self.pilot)

    def handle_new_observation(self, observation: Observation):
        """extract the data and save it to self.data"""
        parsed_data = json.loads(observation.raw)
        self.data = pd.concat(
            [
                self.data,
                pd.DataFrame(
                    {
                        "date_time": [str(parsed_data["time"])],
                        "value": [float(parsed_data["data"])],
                        "id": [str(parsed_data["hash"])],
                    }
                ),
            ],
            ignore_index=True,
        )

    def produce_prediction(self, updated_model=None):
        """
        triggered by
            - model model replaced with a better one
            - new observation on the stream
        """
        updated_model = updated_model or self.stable
        if updated_model is not None:
            forecast = updated_model.predict(stable=self.stable, data=self.data)

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
                    currentValue=self.data,
                    forecast=forecast,
                    observationTime=observationTime,
                    observationHash=observationHash,
                )
                # print("**************************")
                # print(streamforecast)
                # print("**************************")
                self.prediction_produced.on_next(streamforecast)

    def load_data(self) -> pd.DataFrame:
        try:
            return pd.read_csv(
                self.data_path(), names=["date_time", "value", "id"], header=None
            )
        except FileNotFoundError:
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def data_path(self) -> str:
        # print(f"../../data/{generatePathId(streamId=self.streamId)}/aggregate.csv")
        return f"../../data/{generatePathId(streamId=self.streamId)}/aggregate.csv"

    def model_path(self) -> str:
        return f"../../models/{generatePathId(streamId=self.streamId)}"

    def check_observations(self) -> bool:
        """
        Check if the dataframe has fewer than 3 observations.
        Returns:
            bool: True if dataframe has more than 2 rows, False otherwise
        """
        return len(self.data) > 2

    def choose_pipeline(self, inplace: bool = False) -> PipelineInterface:
        """
        everything can try to handle some cases
        Engine
            - low resources available - SKPipeline
            - few observations - SKPipeline
            - (mapping of cases to suitable pipelines)
        examples: StartPipeline, SKPipeline, XGBoostPipeline, ChronosPipeline, DNNPipeline
        """
        if self.check_observations():
            if inplace and not isinstance(self.pilot, SKPipeline):
                self.pilot = SKPipeline() 
            return SKPipeline
        else:
            if inplace and not isinstance(self.pilot, StarterPipeline):
                self.pilot = StarterPipeline()
            return StarterPipeline

    def run(self):
        """
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        Breaks if backtest error stagnates for 3 iterations.
        """
        while True:
            # print(self.pipeline)
            # print(self.pilot)
            trainingResult = self.pilot.fit(data=self.data)
            # print(trainingResult.model)
            if trainingResult.status == 1 and not trainingResult.stagnated:
                if self.pilot.compare(self.stable):
                    if self.pilot.save(self.model_path()):
                        self.stable = copy.deepcopy(self.pilot)
                        self.produce_prediction(self.stable)
            else:
                break

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()
