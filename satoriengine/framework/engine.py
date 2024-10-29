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
            on_next=lambda x: self.handle_new_observation(
                x) if x is not None else None,
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
        streamModel.handle_new_observation(observation)
        if streamModel.thread is None or not streamModel.thread.is_alive():
            streamModel.choose_pipeline(inplace=True)
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
        self.pipeline: PipelineInterface = self.choose_pipeline(getStart=True)
        # the model file itself should tell us what pipeline it is
        # if a model file exists, load it
        # if not create a blank one.
        self.pilot: PipelineInterface = self.pipeline.load(
            self.model_path()) if self.pipeline is not None else None
        self.stable: PipelineInterface = copy.deepcopy(self.pilot)

    def handle_new_observation(self, observation: Observation):
        """extract the data and save it to self.data"""
        parsed_data = json.loads(observation.raw)
        self.data = pd.concat([self.data, pd.DataFrame({
            'date_time': [str(parsed_data['time'])],
            'value': [float(parsed_data['data'])],
            'id': [str(parsed_data['hash'])]
        })], ignore_index=True)

    def produce_prediction(self, updated_model=None):
        """
        triggered by
            - model model replaced with a better one
            - new observation on the stream
        """
        updated_model = updated_model or self.stable
        if updated_model is not None:
            forecast = self.pipeline.predict(
                stable=self.stable, data=self.data)

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

    def load_data(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.data_path(), names=["date_time", "value", "id"], header=None)
        except FileNotFoundError:
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def data_path(self) -> str:
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

    def choose_pipeline(
        self, getStart: bool = False, inplace: bool = False
    ) -> Union[PipelineInterface, None]:
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
            if inplace:
                self.pipeline = None
            return None
        else:
            if self.check_observations():
                if inplace:
                    self.pipeline = SKPipeline
                return SKPipeline
            else:
                if inplace:
                    self.pipeline = StarterPipeline
                return StarterPipeline

    def run(self):
        """
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        Breaks if backtest error stagnates for 3 iterations.
        """
        while True:
            print(self.pipeline)
            trainingResult = self.pilot.fit(data=self.data)
            if trainingResult.status == 1 and not trainingResult.stagnated:
                if self.pilot.compare(self.pilot, trainingResult.model):
                    if self.pilot.save(trainingResult.model, self.model_path()):
                        self.stable = copy.deepcopy(self.pilot)
                        self.produce_prediction(self.stable)
            else:
                break

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()
