# TODO: refactor see issue #24

'''
Basic Reponsibilities of the ModelManager:
1. keep a record of the datasets, features, and parameters of the best model.
2. retrain the best model on new data available, generate and report prediction.
3. continuously generate new models to attempt to find a better one.
    A. search the parameter space smartly
    B. search the engineered feature space smartly
    C. evaluate new datasets when they become available
4. save the best model details and load them upon restart
'''
import time
import pandas as pd
from reactivex.subject import BehaviorSubject
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier
from satorilib import logging
from satorilib.concepts import StreamId
from satorilib.api.disk import Disk
from satorilib.api.interfaces.model import ModelMemoryApi
from satoriengine.concepts import HyperParameter
from satoriengine.model.pilot import PilotModel
from satoriengine.model.stable import StableModel
from typing import Union


class ModelManager:

    config = None

    @classmethod
    def setConfig(cls, config):
        cls.config = config

    def __init__(
        self,
        variable: StreamId,
        output: StreamId,
        targets: list[StreamId] = None,
        memory: ModelMemoryApi = None,
        modelPath: str = None,
        hyperParameters: list[HyperParameter] = None,
        metrics: dict = None,
        features: dict = None,
        chosenFeatures: list[str] = None,
        pinnedFeatures: list[str] = None,
        exploreFeatures: bool = True,
        split: Union[int, float] = .2,
        override: bool = False,
    ):
        '''
        variable: the response variable - that which we are trying to predict. 
                Is a StreamId object which must be entirely specified.
        disk: the disk interface
        memory: the memory interface
        modelPath: the path of the model
        hyperParameters: a list of HyperParameter objects
        metrics: a dictionary of functions that each produce a feature 
                (from 1 dynamic column) example: year over year, rolling average
        features: a dictionary of functions that each take in multiple columns
                of the raw data and ouput a feature (cols known ahead of time)
                example: high minus low, x if y > 1 else 2**z
        chosenFeatures: list of feature names to start with
        pinnedFeatures: list of feature names to keep in model
        exploreFeatures: change features or not
        id: column name of response variable
        split: train test split percentage or count
        override: override the existing model saved to disk if there is one
        '''
        self.variable = variable
        self.output = output
        self.key = self.variable.id
        self.id = self.variable.id
        self.disk = Disk()
        self.memory = memory
        # self.modelPath = modelPath or self.disk.defaultModelPath(self.variable)
        self.targets: list[StreamId] = targets
        self.setupFlags()
        self.get()
        self.stable = StableModel(
            manager=self,
            hyperParameters=hyperParameters or [],
            metrics=metrics,
            features=features or {},
            chosenFeatures=chosenFeatures or [],
            pinnedFeatures=pinnedFeatures or [],
            split=split)
        if not override:
            self.load()
        self.pilot = PilotModel(
            manager=self,
            stable=self.stable,
            exploreFeatures=exploreFeatures)
        # not even necessary right now.
        # self.syncManifest()

    @property
    def prediction(self):
        ''' gets prediction from the stable model '''
        return self.stable.prediction

    def buildStable(self):
        self.stable.build()

    def overview(self):
        def getValues():
            try:
                logging.debug('in overview ', self.data)
                return self.data.dropna().loc[:, (self.variable.source, self.variable.author, self.variable.stream, self.variable.target)].values.tolist()[-20:]
            except Exception as e:
                logging.error('error in overview', e)
                return []

        return {
            'source': self.variable.source,
            'author': self.variable.author,
            'stream': self.variable.stream,
            'target': self.variable.target,
            'value': self.stable.current.values[0][0] if hasattr(self.stable, 'current') else '',
            'prediction': self.stable.prediction if hasattr(self.stable, 'prediction') else '',
            'values': getValues(),
            'predictions': self.stable.predictions if hasattr(self.stable, 'predictions') else [],
            # this isn't the accuracy we really care about (historic accuracy),
            # it's accuracy of this current model on historic data.
            'accuracy': f'{str(self.stableScore*100)[0:5]} %' if hasattr(self, 'stableScore') else '',
            'errs': self.errs if hasattr(self, 'errs') else [],
            'subscribers': 'none'}

    def syncManifest(self):
        # todo: fix this to work with the data from the server
        #       the manifest is only needed for the pilot. server knows the
        #       inputs for the actual stable model. so at this time lest ignore.
        #       model after relay config
        manifest = ModelManager.config.manifest()
        manifest[self.key] = {
            'targets': [x.id for x in self.targets],
            'purged': manifest.get(self.key, {}).get('purged', [])}
        ModelManager.config.put('manifest', data=manifest)

    ### FLAGS ################################################################################

    def setupFlags(self):
        self.modelUpdated = BehaviorSubject(None)
        self.variableUpdated = BehaviorSubject(None)
        self.targetUpdated = BehaviorSubject(None)
        self.inputsUpdated = BehaviorSubject(None)
        self.predictionUpdate = BehaviorSubject(None)
        self.predictionEdgeUpdate = BehaviorSubject(None)
        self.newAvailableInput = BehaviorSubject(None)

    ### GET DATA ####################################################################

    # @staticmethod
    # def addFeatureLevel(df:pd.DataFrame):
    #    ''' adds a feature level to the multiindex columns'''
    #    return pd.MultiIndex.from_tuples([c + ('Raw',)  for c in df.columns])

    def get(self):
        ''' gets the raw data from disk '''

        def handleEmpty():
            '''
            todo: what should we do if no data available yet? 
            should self.data be None? or should it be an empty dataframe without our target columns?
            or should it be an empty dataframe with our target columns?
            It seems like it should just be None and that we should halt behavior until it has a
            threshold amount of data.
            '''
            self.data = self.data if self.data is not None else pd.DataFrame(
                {x.key: [] for x in set(self.targets)})

        self.data = self.disk.gather(
            streamIds=self.targets,
            targetColumn=self.id)
        logging.debug('SETTING DATA:')
        logging.debug('self.targets', self.targets)
        logging.debug('self.id', self.id)
        logging.debug('self.data', self.data)
        handleEmpty()

    ### FEATURE DATA ####################################################################

    def showFeatureData(self):
        '''
        returns true raw feature importance
        example: {
            'Close': 0.6193444132804871,
            'High': 0.16701968474080786,
            'Low': 0.38159190578153357}
        '''
        return self.stable.showFeatureData()

    ### META TRAIN ######################################################################

    def evaluateCandidate(self):
        ''' 
        model consists of the hyperParameter values and the chosenFeatures, 
        these are saved to disk. we also replace the model object itself.
        '''
        def scoreRegressiveModels():
            '''
            MAE of 0: 
                This would indicate a perfect model, meaning the predictions
                perfectly match the true values. However, achieving an MAE of
                exactly 0 is rare and often unlikely in practical scenarios.
            Small MAE: 
                A smaller MAE indicates that the model's predictions are, on
                average, closer to the true values. The closer the MAE is to 0,
                the better the model's performance. However, what is considered
                a "small" MAE varies depending on the problem and context.
            MAE relative to the scale of the target variable:
                The interpretation of MAE should be considered relative to the
                scale of your target variable. If the target variable has a
                large range or high variance, a relatively small MAE might still
                represent a good model performance.
            Comparison between models:
                MAE is often more interpretable and intuitive when used for
                comparing different models. If you have multiple models, you can
                compare their MAE values directly. A model with a lower MAE is
                generally considered to have better predictive performance
            '''
            self.stableScore = mean_absolute_error(
                self.stable.testY,
                self.stable.xgb.predict(self.stable.testX))
            self.pilotScore = mean_absolute_error(
                self.pilot.testY,
                self.pilot.xgb.predict(self.pilot.testX))
            result = self.pilotScore < self.stableScore
            if result:
                logging.debug(self.variable.stream, self.variable.target, 'scores:',
                              self.stableScore, self.pilotScore)
                # maybe this should be done on broadcast? saving it to memory
                if not hasattr(self, 'errs'):
                    self.errs = []
                self.errs.append(self.pilotScore)
            return result

        def scoreClassificationModels():
            ''' 
            The R2 score typically ranges from -∞ to 1. 
            1 indicates a perfect fit.
            0 indicates the model performs no better than randomly guessing.
            negative values indicate that the model's predictions are worse than
            simply guessing the mean of the target variable.
            '''
            self.stableScore = self.stable.xgb.score(
                self.stable.testX,
                self.stable.testY)
            self.pilotScore = self.pilot.xgb.score(
                self.pilot.testX,
                self.pilot.testY)
            return self.pilotScore > self.stableScore

        def replaceStableModel():
            for param in self.stable.hyperParameters:
                # is this right? it looks right but I don't think the stable model ever updates from the pilot
                param.value = param.test
            self.stable.chosenFeatures = self.pilot.testFeatures
            self.stable.featureSet = self.pilot.testFeatureSet
            self.stable.xgb = self.pilot.xgb
            self.save()
            return True

        if isinstance(self.stable.xgb, XGBRegressor):
            if scoreRegressiveModels():
                return replaceStableModel()
        elif isinstance(self.stable.xgb, XGBClassifier):
            if scoreClassificationModels():
                return replaceStableModel()
        return False

    ### SAVE ###########################################################################

    def save(self):
        ''' save the current model '''
        self.disk.saveModel(
            self.stable.xgb,
            # modelPath=self.modelPath,
            streamId=self.variable,
            hyperParameters=self.stable.hyperParameters,
            chosenFeatures=self.stable.chosenFeatures)

    def load(self):  # -> bool:
        ''' loads the model - happens on init so we automatically load our progress '''
        xgb = self.disk.loadModel(
            # modelPath=self.modelPath,
            streamId=self.variable,
        )
        logging.debug('LOADING STABLE', xgb)
        if xgb == False:
            return False
        if (
            all([scf in self.stable.features.keys() for scf in xgb.savedChosenFeatures]) and
            # all([shp in self.stable.hyperParameters for shp in xgb.savedHyperParameters])
            True
        ):
            self.stable.xgb = xgb
            self.stable.hyperParameters = xgb.savedHyperParameters
            self.stable.chosenFeatures = xgb.savedChosenFeatures
        return True

    ### LIFECYCLE ######################################################################

    def runPredictor(self):
        def makePrediction(isVariable=False):
            logging.debug('in makePrediction')
            if isVariable and self.stable.build():
                logging.debug('PRODUCING PREDICITON')
                self.stable.producePrediction()
                show(
                    f'prediction - {self.variable.stream} {self.variable.target}:', self.stable.prediction)
                logging.debug('BROADCASTING PREDICITON')
                self.predictionUpdate.on_next(self)
            # this is a feature to be added - a second publish stream which requires a
            # different dataset - one where the latest update is taken into account.
            #    if self.edge:
            #        self.predictionEdgeUpdate.on_next(self)
            # elif self.edge:
            #    self.stable.build()
            #    self.predictionEdge = self.producePrediction()
            #    self.predictionEdgeUpdate.on_next(self)

        def makePredictionFromNewModel():
            show(f'model updated - {self.variable.stream} {self.variable.target}:',
                 f'{self.stableScore}, {self.pilotScore}')
            makePrediction()

        def makePredictionFromNewInputs():
            '''
            go get the entire dataset from memory. this is triggered at the end
            of gathering and merging the ipfs history data.
            '''
            self.get()
            makePrediction()

        def makePredictionFromNewTarget(incremental):
            # note: on disk we remove all duplicates, but in the model, datasets
            #       are combined and often have duplicates because they show
            #       what the data was in the interim. However, we probably still
            #       don't want to generally blindly append a new row as it might
            #       be an entire duplicate row. (each row should have at least
            #       one change somewhere unless the model uses an algorithm that
            #       wants it at constant time intervals or something). so we
            #       remove duplicates here instead of in self.memory because
            #       it's really the model's perogative.
            self.data = self.memory.dropDuplicates(
                self.memory.appendInsert(
                    df=self.data,
                    incremental=incremental))
            makePrediction()

        def makePredictionFromNewVariable(incremental):
            logging.debug('in makePredictionFromNewVariable')
            for col in incremental.columns:
                if col not in self.data.columns:
                    incremental = incremental.drop(col, axis=1)
            # incremental.columns = ModelManager.addFeatureLevel(df=incremental)
            self.data = self.memory.dropDuplicates(
                self.memory.appendInsert(
                    df=self.data,
                    incremental=incremental))
            makePrediction(isVariable=True)

        self.modelUpdated.subscribe(
            lambda x: makePredictionFromNewModel() if x is not None else None)
        self.inputsUpdated.subscribe(
            lambda x: makePredictionFromNewInputs() if x else None)
        self.variableUpdated.subscribe(
            lambda x: makePredictionFromNewVariable(x) if x is not None else None)
        self.targetUpdated.subscribe(
            lambda x: makePredictionFromNewTarget(x) if x is not None else None)

    def runExplorer(self):
        if hasattr(self.stable, 'target') and hasattr(self.stable, 'xgbStable'):
            try:
                self.pilot.build()
                if self.evaluateCandidate():
                    self.modelUpdated.on_next(True)
            except NotFittedError as e:
                '''
                this happens on occasion...
                maybe making  self.xgbStable a deepcopy would fix
                '''
                logging.debug('not fitted', e)
            # except AttributeError as e:
            #    '''
            #    this happens at the beginning of running when we have not set
            #    self.xgbStable yet.
            #    '''
            #    #logging.debug('Attribute', e)
            #    pass
            except Exception as e:
                logging.debug('UNEXPECTED', e)
        else:
            time.sleep(1)

    def syncAvailableInputs(self):

        def sync(x):
            '''
            add the new datastreams and histories to the top 
            of the list of things to explore and evaluate 
            '''
            # something like this?
            # self.features.append(x)
            #
            # self.targets.append(StreamId(x))  or something
            # self.syncManifest()  then sync manifest when you change targets.
            # maybe remove targets that aren't being used as any features.. somewhere?

        self.newAvailableInput.subscribe(
            lambda x: sync(x) if x is not None else None)


# testing
def show(name, value):
    if isinstance(value, pd.DataFrame):
        print(f'\n{name}\n', value.tail(2))
    else:
        print(f'\n{name}\n', value)
