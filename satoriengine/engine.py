from typing import Optional
import threading
import pandas as pd
from satorilib.concepts.structs import StreamId
from satoriengine.managers.data import DataManager
from satoriengine.managers.model import ModelManager
from satoriengine.view import View


class Engine:

    def __init__(
        self,
        start=None,
        data: DataManager = None,
        models: set[ModelManager] = None,
        api: Optional[object] = None,
        view: View = None,
    ):
        '''
        data - a DataManager for the data
        model - a ModelManager for the model
        models - a list of ModelManagers
        '''
        self.start = start
        self.data = data
        self.models = models
        self.view = view
        self.api = api

    def getObservationOf(
        self,
        streamId: StreamId,
        timestamp: str
    ) -> tuple[str, str]:
        '''
        queries that models to find the observation prior to the timestamp of a
        given stream. returns the timestamp and the value of the observation.
        if there is no observation before the given timestamp, it returns 
        (None, None) if it has a problem getting the data it returns an empty
        tuple (,).
        '''
        df = getDataOfBefore(streamId, timestamp)
        if df is None:
            return (,)
        try:
            df = df.sort_index()
            return df.loc[df.index < target_timestamp].iloc[-1]
        except Exception as _:
            return (None, None)
    

    def getObservationCountOf(
        self,
        streamId: StreamId,
        timestamp: str
    ) -> Union[int, None]:
        '''
        queries that models to find the count of observations prior to the 
        timestamp of a given stream. returns the count. if there are no 
        observations before the given timestamp, it returns 0, if it has a 
        problem getting the data it returns None.
        '''
        df = getDataOfBefore(streamId, timestamp)
        if df is None:
            return None
        return df.shape[0]
    
        # TODO NEXT:
        #
        # after this is done we can add the api to the protocol for asking for
        # counts, etc.

    def getDataOfBefore(
        self,
        streamId: StreamId,
        timestamp: str,
    ) -> pd.DataFrame:
        '''
        queries that models to find the dataframe of observations prior to the 
        timestamp of a given stream. returns the df. if there are no 
        observations before the given timestamp, it returns an empty dataframe,
        if it has a problem getting the data it returns None.
        '''
        # TODO NEXT:
        # loop through models and find the one that has the stream
        # get the data of the model and
        # filter down to index and the column that corresponds to the stream
        # dedupe that subset dataframe
        # subset the dataframe to only those rows prior to the timestamp
        # return df

    def out(self, predictions, scores, data=True, model=None):
        ''' old functionality that must be accounted for in new design
        if self.view is not None:
            self.view.print(**(
                {
                    'Predictions:\n':predictions,
                    '\nScores:\n':scores
                } if data else {model.id: 'loading... '}))
        '''
        self.view.print(**(
            {
                'Predictions:\n': predictions,
                '\nScores:\n': scores
            } if data else {model.id: 'loading... '}))

    def updateView(self, predictions, scores):
        '''
        old functionality that must be accounted for in new design
        non-reactive jupyter view
        predictions[model.id] = model.producePrediction()
        scores[model.id] = f'{round(stable, 3)} ({round(test, 3)})'
        inputs[model.id] = model.showFeatureData()
        if first or startingPredictions != predictions:
            first = False
            if self.api is not None:
                self.api.send(model, predictions, scores)
            if self.view is not None:
                self.view.view(model, predictions, scores)
                out()
        out(data=False)
        '''
        if self.view is not None:
            self.view.view(self.data, predictions, scores)
            # out()
        # out(data=False)

    def run(self):
        ''' Main '''

        def publisher():
            '''
            publishes predictions on demand
            this should probably be broken out into a service
            that creates a stream and a service that publishes...
            '''
            self.data.runPublisher(self.models)

        def subscriber():
            '''
            listens for external updates on subscriptions - 
            turn this into a stream rather than a loop - 
            triggered from flask app.
            this should probably be broken out into a service
            that subscribes and a service that listens...
            should be on demand
            '''
            self.data.runSubscriber(self.models)

        # # not used yet
        # def scholar():
        #    ''' always looks for external data and compiles it '''
        #    while True:
        #        time.sleep(1)
        #        if not self.start.paused:
        #            self.data.runScholar(self.models)

        def predictor(model: ModelManager):
            ''' produces predictions on demand '''
            model.runPredictor()

        # # not used yet
        # def sync(model: ModelManager):
        #    ''' sync available inputs found and compiled by scholar on demand '''
        #    model.syncAvailableInputs()

        def explorer(model: ModelManager):
            ''' always looks for a better model '''
            while True:
                if not self.start.paused:
                    model.runExplorer()

        def watcher(model: ModelManager):
            ''' for reactive views... '''
            if self.view:
                self.view.listen(model)

        publisher()
        subscriber()
        threads = {}
        # threads['scholar'] = threading.Thread(target=scholar, daemon=True)
        for model in self.models:
            # we have to run this once for each model to complete its initialization
            model.buildStable()
            predictor(model)
            # sync(model)
            if self.view and self.view.isReactive:
                watcher(model)
            threads[f'{model.id}.explorer'] = threading.Thread(
                target=explorer, args=[model], daemon=True)
        for thread in threads.values():
            thread.start()


howToRun = '''
# see example notebook
'''
