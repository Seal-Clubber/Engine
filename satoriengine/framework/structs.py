import datetime as dt
import pandas as pd
from satorilib.concepts import StreamId


class StreamForecast:
    def __init__(
        self,
        streamId: StreamId,
        forecast: pd.DataFrame,
        observationTime: str,
        observationHash: str,
    ):
        self.streamId = streamId
        self.forecast = forecast
        self.observationTime = observationTime
        self.observationHash = observationHash

    def firstPrediction(self):
        return StreamForecast.firstPredictionOf(self.forecast)

    @staticmethod
    def firstPredictionOf(forecast: pd.DataFrame):
        return forecast["pred"].iloc[0]
