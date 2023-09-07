''' interface for a StableModel '''
from typing import Union
from satoriengine.concepts import HyperParameter
#from satoriengine.managers.model import ModelManager


class StableModelInterface:

    def __init__(self,
                 manager: 'ModelManager',
                 # used in pilot and manager
                 hyperParameters: list[HyperParameter] = None,
                 metrics: dict = None,
                 features: dict = None,  # used in pilot and manager
                 # used in pilot and manager
                 chosenFeatures: list[str] = None,
                 pinnedFeatures: list[str] = None,  # used in pilot
                 split: Union[int, float] = .2  # used in pilot
                 ):
        '''
        manager: parent object
        hyperParameters: a list of HyperParameter objects
        metrics: a dictionary of functions that each produce
                    a feature (from 1 dynamic column)
                    example: year over year, rolling average
        features: a dictionary of functions that each take in
                    multiple columns of the raw data and ouput
                    a feature (cols known ahead of time)
                    example: high minus low, x if y > 1 else 2**z
        chosenFeatures: list of feature names to start with
        pinnedFeatures: list of feature names to keep in model
        split: train test split percentage or count
        '''
        self.manager = manager  # use a setter?
        self.hyperParameters = hyperParameters or []
        self.chosenFeatures = chosenFeatures or []
        self.pinnedFeatures = pinnedFeatures or []
        self.features = features or {}
        self.metrics = metrics
        self.split = split
        self.featureData = {}
        self.xgbInUse = False
        self.xgb = None
        self._produceFeatureStructure()
        if not self.data.empty:
            self._produceFeatureSet()

    @property
    def data(self):
        ''' gets data from the model manager '''
        return self.manager.data

    @property
    def id(self):
        ''' gets id from the model manager '''
        return self.manager.id

    ### FEATURES ####################################################################

    def _produceFeatureStructure(self):
        ''' generates self.features '''

    def _produceFeatureSet(self):
        ''' produces a feature set for the stable model '''

    def leastValuableFeature(self):
        ''' returns the least valuable feaure '''

    ### FEATURE DATA ####################################################################

    def showFeatureData(self):
        '''
        returns true raw feature importance
        example: {
            'Close': 0.6193444132804871,
            'High': 0.16701968474080786,
            'Low': 0.38159190578153357}
        '''

    ### CURRENT ####################################################################

    def producePrediction(self):
        '''generates a prediction'''

    ### MAIN PROCESSES #################################################################

    def build(self):
        '''builds stable model'''
