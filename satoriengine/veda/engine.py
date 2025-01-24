from satoriengine.veda.adapters import ModelAdapter, SKAdapter, StarterAdapter, XgbAdapter, XgbChronosAdapter
from satoriengine.veda.data import StreamForecast, cleanse_dataframe, validate_single_entry
from satorilib.logging import INFO, setup, debug, info, warning, error
from satorilib.disk.filetypes.csv import CSVManager
from satorilib.concepts import Stream, StreamId, StreamUuid, Observation
from satorilib.disk import getHashBefore
from satorilib.utils.system import getProcessorCount
from satorilib.utils.time import datetimeToTimestamp, now
from satorilib.utils.hash import hashIt, generatePathId
from satorilib.datamanager import DataClient, PeerInfo, Message, Subscription
from satorineuron import config
from reactivex.subject import BehaviorSubject
import pandas as pd
import numpy as np
import threading
import json
import copy
import time
import os
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

setup(level=INFO)


class Engine:

    @classmethod
    async def create(cls) -> 'Engine':
        engine = cls()
        await engine.initialize()
        return engine
    
    def __init__(self):
        self.streamModels: Dict[StreamId, StreamModel] = {}
        self.newObservation: BehaviorSubject = BehaviorSubject(None)
        self.predictionProduced: BehaviorSubject = BehaviorSubject(None)
        self.subcriptions: dict[str, PeerInfo] = {}
        self.publications: dict[str, PeerInfo] = {}
        self.dataServerIp: str = ''
        self.dataClient: DataClient = DataClient()
        self.paused: bool = False
        self.threads: list[threading.Thread] = []
    
    def handlePrediction(self, predictionInputs: PredictionInputs):

        def registered(pubkey: str):
            if pubkey is not None:
                if pubkey not in self.wallets.keys():
                    walletId = Wallet.getIdFromPubkey(pubkey=pubkey)
                    if walletId is not None:
                        self.wallets[pubkey] = walletId
                    return walletId
                return self.wallets[pubkey]
            return None

        if auth(predictionInputs.walletPayload):
            walletId = registered(predictionInputs.walletPubkey)
            if walletId is not None:
                predictionInputs.setWalletId(walletId=walletId)
                # just record record those that are staked
                # df = database.get.getStake(wallet_id=walletId)
                # if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                self.recordPredictionInDatabase(predictionInputs)

    def processQueue(self):
        while True:
            try:
                self.handlePrediction(self.queue.get())
            except Exception as e:
                self.logging.error(f"Error saving to database: {str(e)}")
            finally:
                # self.queue.task_done()
                pass

    # could use async task instead if we want
    def startProcessing(self):
        recordPredictionThread = threading.Thread(
            target=self.processQueue,
            daemon=True)
        recordPredictionThread.start()



    async def initialize(self):
        await self.connectToDataServer()
        await self.getPubSubInfo()
        # setup subscriptions to external dataservers
        # on observation 
        #   pass to data server (for it to save to disk)
        #   pass to handle observation
        #   make sure we update training data
        # self.setupSubscriptions()
        await self.initializeModels()

    def pause(self, force: bool = False):
        if force:
            self.paused = True
        for streamModel in self.streamModels.values():
            streamModel.pause()

    def resume(self, force: bool = False):
        if force:
            self.paused = False
        if not self.paused:
            for streamModel in self.streamModels.values():
                streamModel.resume()

    async def connectToDataServer(self):
        self.dataServerIp = config.get().get('server ip', '0.0.0.0')
        try:
            await self.dataClient.connectToServer(peerHost=self.dataServerIp)
            info("Successfully connected to Server at :", self.dataServerIp, color="green")
        except Exception as e:
            error("Error connecting to server : ", e)
            self.dataServerIp = self.start.server.getPublicIp().text.split()[-1] # TODO : is this correct?

    async def getPubSubInfo(self):
        try:
            pubsubMap = await self.dataClient.sendRequest(peerHost=self.dataServerIp, method='get-pubsub-map')
            for sub_uuid, data in pubsubMap.streamInfo.items():
                self.subcriptions[sub_uuid] = PeerInfo(data['subscription_subscribers'], data['subscription_publishers'])
                self.publications[data['publication_uuid']] = PeerInfo(data['publication_subscribers'], data['publication_publishers'])
            print("Inside the Engine", len(self.subcriptions))
        except Exception as e:
            error(f"Failed to send request {e}")
    
    def setupSubscriptions(self):
        self.newObservation.subscribe(
            on_next=lambda x: self.handleNewObservation(
                x) if x is not None else None,
            on_error=lambda e: self.handleError(e),
            on_completed=lambda: self.handleCompletion())

    async def initializeModels(self):
        for subuuid, pubuuid in zip(self.subcriptions.keys(), self.publications.keys()): 
            peers = self.subcriptions[subuuid]
            self.streamModels[subuuid] = await StreamModel.create(
                streamId=subuuid,
                predictionStreamId=pubuuid,
                serverIp=self.dataServerIp,
                peerInfo=peers,
                dataClient=self.dataClient,
                predictionProduced=self.predictionProduced)
            self.streamModels[subuuid].chooseAdapter(inplace=True)
            self.streamModels[subuuid].run_forever()

    def handleNewObservation(self, observation: Observation):
        # spin off a new thread to handle the new observation
        thread = threading.Thread(
            target=self.handleNewObservationThread,
            args=(observation,))
        thread.start()
        self.threads.append(thread)
        self.cleanupThreads()

    def cleanupThreads(self):
        for thread in self.threads:
            if not thread.is_alive():
                self.threads.remove(thread)
        debug(f'prediction thread count: {len(self.threads)}')

    def handleNewObservationThread(self, observation: Observation):
        streamModel = self.streamModels.get(observation.streamId)
        if streamModel is not None:
            self.pause()
            streamModel.handleNewObservation(observation)
            if streamModel.thread is None or not streamModel.thread.is_alive():
                streamModel.chooseAdapter(inplace=True)
                streamModel.run_forever()
            if streamModel is not None:
                info(
                    f'new observation, making prediction using {streamModel.adapter.__name__}', color='blue')
                streamModel.producePrediction()
            self.resume()

    def handleError(self, error):
        print(f"An error occurred new_observaiton: {error}")

    def handleCompletion(self):
        print("newObservation completed")


class StreamModel:

    @classmethod
    async def create(
        cls, 
        streamId: StreamUuid,
        predictionStreamId: StreamUuid,
        serverIp: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        predictionProduced: BehaviorSubject
    ):
        streamModel = cls(
            streamId,
            predictionStreamId,
            serverIp,
            peerInfo,
            dataClient,
            predictionProduced
        )
        await streamModel.initialize()
        return streamModel

    def __init__(
        self,
        streamId: StreamUuid,
        predictionStreamId: StreamUuid,
        serverIp: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        predictionProduced: BehaviorSubject,
    ):
        self.cpu = getProcessorCount()
        self.preferredAdapters: list[ModelAdapter] = [StarterAdapter, XgbAdapter, XgbChronosAdapter]# SKAdapter #model[0] issue
        self.defaultAdapters: list[ModelAdapter] = [XgbAdapter, XgbAdapter, StarterAdapter]
        self.failedAdapters = []
        self.thread: threading.Thread = None
        self.streamId: StreamUuid = streamId
        self.predictionStreamId: StreamUuid = predictionStreamId
        self.serverIp = serverIp
        self.peerInfo: PeerInfo = peerInfo
        self.dataClient: DataClient = dataClient
        self.predictionProduced: BehaviorSubject = predictionProduced
        self.rng = np.random.default_rng(37)
        self.publisherHost = self.peerInfo.publishersIp[0]

    async def initialize(self):
        self.data: pd.DataFrame = await self.loadData()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamId)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        debug(f'AI Engine: stream id {self.streamId} using {self.adapter.__name__}', color='teal')

    async def init2(self):
        # DO LATER: for loop for all the streams we want to subscribe to (raw data stream and all feature streams)
        self.publisherHost = await self.connectToPeer()
        await self.syncData()
        await self.makeSubscription()
        # self.listenToSubscription()

    async def connectToPeer(self):
        # choose peer to connect to
        # - connect to a peer for a stream
        #     - attempt connection to the source first (publisher)
        #     - if able to connect, make sure they have the stream we're looking for available for subscribing to
        #     - elif not: 
        #       - handle subscriber list
        #         - filter our own ip out of the subscriber list
        #         - randomize subscriber list (shuffle payload[uuid][1:])
        #     - go down the subscriber list until you find one...


        async def _isPublisherActive(publisherIp: str) -> bool:
            async def _isActive(publisherIp):
                # TODO : Logic to check if the publisher is active
                pass

            if _isActive(publisherIp):
                return True
            return False

        try:
            if _isPublisherActive(self.publisherHost):
                return self.publisherHost
            else:
                # update the server that self.peerInfo.publishersIp[0] is not active and remove it from its list.
                await self._removePublisher(self.publisherHost)
                self.peerInfo.subscribersIp = [
                    ip for ip in self.peerInfo.subscribersIp if ip != self.serverIp
                ]
                self.rng.shuffle(self.peerInfo.subscribersIp)
                for subscriberIp in self.peerInfo.subcribersIp:
                    if _isPublisherActive(subscriberIp):
                        await self._addPublisher(subscriberIp)
                        return subscriberIp

                # try to connect to a subscriber until we find one
                # must ask the other subscriber that we connect to if they have
                # an active connection to the data (including it's own publsihed
                # data, which it can assume is active) - make endpoint on the 
                # DataServer and it must get that information from one of it's 
                # local clients (meaning neuron or Engine data clients)
                error("Publisher does not contain subscription stream")
        except Exception as e:
            error("Error, cannot connect to Publisher : ", e)
        
        # when finally successful we must tell our own dataserver I am
        # subscribed to x and will pass it's observations to you.

    async def syncData(self):
        '''
        - this can be highly optimized. but for now we do the simple version
        - just ask for their entire dataset every time
            - if it's different than the df we got from our own dataserver,
              then tell dataserver to save this instead
            - replace what we have
        '''
        try:
            externalDataJson = await self.dataClient.sendRequest(
                peerHost=self.publisherHost, 
                uuid=self.streamId,
                method='stream-info',
            )
            externalDf = pd.read_json(externalDataJson.data, orient='split')
        except Exception as e:
            error("Error cannot connect to peer: ", e)

        if not externalDf.equals(self.data) and len(externalDf) > 0:
            self.data = externalDf
            try:
                await self.dataClient.sendRequest(
                    peerHost=self.serverIp,
                    uuid=self.streamId,
                    method='insert',
                    data=externalDf,
                    replace=True,
                )
            except Exception as e:
                error("Error cannot connect to Server: ", e)

    async def makeSubscription(self):
        '''
        - and subscribe to the stream so we get the information
            - whenever we get an observation on this stream, pass to the DataServer
        - continually generate predictions for prediction publication streams and pass that to 
        '''
        # for every stream we care about - raw data stream, and all supporting streams
        await self.dataClient.subscribe(
              uuid=self.streamId,
              callback=self.handleSubscriptionMessage)

    async def handleSubscriptionMessage(self, subscription: Subscription, message: Message, updatedModel=None):
        pass

    def listenToSubscription():
        '''
        some messages will be on our response variable (subscription.uuid = raw data stream uuid)
         - append to the data (handleNewObservation)
         - produce prediction if we can (producePrediction)
         - pass prediction to our server (create function or add call to server to end of producePrediction)
        other message will be on features (subscription.uuid != raw data stream uuid)
         - append to the data

        callbacks could just start a thread to do these things.
        '''
        pass

    async def _addPublisher(self, publisherIp):
        ''' adds the publisher ip to server '''
        await self.dataClient.sendRequest(
                peerHost=self.serverIp,
                uuid={self.streamId: publisherIp},
                method="add-publisherIp"
            )
        
    async def _removePublisher(self, publisherIp):
        ''' removes the publisher ip from server '''
        await self.dataClient.sendRequest(
                peerHost=self.serverIp,
                uuid={self.streamId: publisherIp},
                method="remove-publisherIp"
            )

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def handleNewObservation(self, observation: Observation):
        """extract the data and save it to self.data"""
        parsedData = json.loads(observation.raw)
        if validate_single_entry(parsedData["time"], parsedData["data"]):
            self.data = pd.concat(
                [
                    self.data,
                    pd.DataFrame({
                        "date_time": [str(parsedData["time"])],
                        "value": [float(parsedData["data"])],
                        "id": [str(parsedData["hash"])]}),
                ],
                ignore_index=True)
        else:
            error("Row not added due to corrupt observation")

    
    # async producePrediction(self, subscription: Subscription, message: Message, updatedModel=None):
    # make this async
    def producePrediction(self, updatedModel=None):
        """
        triggered by
            - model replaced with a better one
            - new observation on the stream
        """
        try:
            updatedModel = updatedModel or self.stable
            if updatedModel is not None:
                forecast = updatedModel.predict(data=self.data)
                if isinstance(forecast, pd.DataFrame):
                    observationTime = datetimeToTimestamp(now())
                    prediction = StreamForecast.firstPredictionOf(forecast)
                    # observationHash = hashIt(
                    #     getHashBefore(pd.DataFrame(), observationTime)
                    #     + str(observationTime)
                    #     + str(prediction))
                    observationHash = 'random' # TODO : new hashing method
                    self.save_prediction(
                        observationTime, prediction, observationHash)
                    streamforecast = StreamForecast(
                        streamId=self.streamId,
                        predictionStreamId=self.predictionStreamId,
                        currentValue=self.data,
                        forecast=forecast,  # maybe we can fetch this value from predictionHistory
                        observationTime=observationTime,
                        observationHash=observationHash,
                        predictionHistory=CSVManager().read(self.prediction_data_path()))
                    self.predictionProduced.on_next(streamforecast)
                else:
                    raise Exception("Forecast not in dataframe format")
        except Exception as e:
            error(e)
            self.fallback_prediction()

    def fallback_prediction(self):
        if os.path.isfile(self.modelPath()):
            try:
                os.remove(self.modelPath())
                debug("Deleted failed model file", color="teal")
            except Exception as e:
                error(f"Failed to delete model file: {str(e)}")
        backupModel = self.defaultAdapters[-1]()
        try:
            trainingResult = backupModel.fit(data=self.data)
            if abs(trainingResult.status) == 1:
                self.producePrediction(backupModel)
        except Exception as e:
            error(f"Error training new model: {str(e)}")

    def save_prediction(
        self,
        observationTime: str,
        prediction: float,
        observationHash: str,
    ) -> pd.DataFrame:
        # alternative - use data manager: self.predictionUpdate.on_next(self)
        df = pd.DataFrame(
            {"value": [prediction], "hash": [observationHash]},
            index=[observationTime])
        df.to_csv(
            self.prediction_data_path(),
            float_format="%.10f",
            mode="a",
            header=False)
        return df

    async def loadData(self) -> pd.DataFrame:
        try:
            datasetJson = await self.dataClient.sendRequest(
                    peerHost=self.serverIp, 
                    uuid=self.streamId,
                    method="stream-data"
                    )
            df = pd.read_json(datasetJson.data, orient='split')
            output_path = os.path.join('csvs', f'{self.streamId}.csv')
            df.to_csv(output_path, index=False)
            return df
        except Exception:
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def prediction_data_path(self) -> str:
        return (
            '/Satori/Neuron/data/testprediction/'
            f'{self.predictionStreamId}.csv')

    def modelPath(self) -> str:
        return (
            '/Satori/Neuron/models/veda/'
            f'{self.streamId}/'
            f'{self.adapter.__name__}.joblib')

    def chooseAdapter(self, inplace: bool = False) -> ModelAdapter:
        """
        everything can try to handle some cases
        Engine
            - low resources available - SKAdapter
            - few observations - SKAdapter
            - (mapping of cases to suitable adapters)
        examples: StartPipeline, SKAdapter, XGBoostPipeline, ChronosPipeline, DNNPipeline
        """
        # TODO: this needs to be aultered. I think the logic is not right. we
        #       should gather a list of adapters that can be used in the
        #       current condition we're in. if we're already using one in that
        #       list, we should continue using it until it starts to make bad
        #       predictions. if not, we should then choose the best one from the
        #       list - we should optimize after we gather acceptable options.

        if False: # for testing specific adapters
            adapter = XgbChronosAdapter
        else:
            import psutil
            availableRamGigs = psutil.virtual_memory().available / 1e9
            adapter = None
            for p in self.preferredAdapters:
                if p in self.failedAdapters:
                    continue
                if p.condition(data=self.data, cpu=self.cpu, availableRamGigs=availableRamGigs) == 1:
                    adapter = p
                    break
            if adapter is None:
                for adapter in self.defaultAdapters:
                    if adapter not in self.failedAdapters:
                        break
                if adapter is None:
                    adapter = self.defaultAdapters[-1]
        if (
            inplace and (
                not hasattr(self, 'pilot') or
                not isinstance(self.pilot, adapter))
        ):
            info(
                f'AI Engine: stream id {self.streamId} '
                f'switching from {self.adapter.__name__} '
                f'to {adapter.__name__} on {self.streamId}',
                color='teal')
            self.adapter = adapter
            self.pilot = adapter(uid=self.streamId)
            self.pilot.load(self.modelPath())
        return adapter

    def run(self):
        """
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        Breaks if backtest error stagnates for 3 iterations.
        """
        while len(self.data) > 0:
            if self.paused:
                time.sleep(10)
                continue
            self.chooseAdapter(inplace=True)
            try:
                trainingResult = self.pilot.fit(data=self.data, stable=self.stable)
                if trainingResult.status == 1:
                    if self.pilot.compare(self.stable):
                        if self.pilot.save(self.modelPath()):
                            self.stable = copy.deepcopy(self.pilot)
                            info(
                                "stable model updated for stream:",
                                self.streamId,
                                print=True)
                            self.producePrediction(self.stable)
                else:
                    debug(f'model training failed on {self.streamId} waiting 10 minutes to retry')
                    self.failedAdapters.append(self.pilot)
                    time.sleep(60*10)
            except Exception as e:
                import traceback
                traceback.print_exc()
                error(e)
                try:
                    import numpy as np
                    print(self.pilot.dataset)
                except Exception as e:
                    pass

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()
