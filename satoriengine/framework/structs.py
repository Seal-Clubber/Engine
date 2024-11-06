import datetime as dt
import pandas as pd
from satorilib.concepts import StreamId


class StreamForecast:
    def __init__(
        self,
        streamId: StreamId,
        currentValue: pd.DataFrame,
        forecast: pd.DataFrame,
        observationTime: str,
        observationHash: str,
    ):
        self.streamId = streamId
        self.currentValue = currentValue
        self.forecast = forecast
        self.observationTime = observationTime
        self.observationHash = observationHash

    def firstPrediction(self):
        return StreamForecast.firstPredictionOf(self.forecast)
    
    def latestValue(self):
        return StreamForecast.latestValue(self.currentValue)

    @staticmethod
    def firstPredictionOf(forecast: pd.DataFrame):
        return forecast["pred"].iloc[0]
    
    @staticmethod
    def latestValueOf(forecast: pd.DataFrame):
        return forecast["value"].iloc[-1]
