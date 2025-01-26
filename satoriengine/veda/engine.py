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
import asyncio
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
            streamModel.appendNewData(observation)
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
        streamUuid: str,
        predictionStreamUuid: str,
        serverIp: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        predictionProduced: BehaviorSubject
    ):
        streamModel = cls(
            streamUuid,
            predictionStreamUuid,
            serverIp,
            peerInfo,
            dataClient,
            predictionProduced
        )
        await streamModel.initialize()
        return streamModel

    def __init__(
        self,
        streamUuid: str,
        predictionStreamUuid: str,
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
        self.streamUuid: str = streamUuid
        self.predictionStreamUuid: str = predictionStreamUuid
        self.serverIp = serverIp
        self.peerInfo: PeerInfo = peerInfo
        self.dataClient: DataClient = dataClient
        self.predictionProduced: BehaviorSubject = predictionProduced
        self.rng = np.random.default_rng(37)
        self.publisherHost = self.peerInfo.publishersIp[0]
        self.isConnectedToPeer = False

    async def initialize(self):
        self.data: pd.DataFrame = await self.loadData()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamUuid)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        debug(f'AI Engine: stream id {self.streamUuid} using {self.adapter.__name__}', color='teal')

    async def init2(self):
        # DO LATER: for loop for all the streams we want to subscribe to (raw data stream and all feature streams)
        self.isConnectedToPeer = await self.connectToPeer()
        if self.isConnectedToPeer:
            await self._addStream()
            await self.syncData()
            await self.makeSubscription()
            await self.listenToSubscription() # the failure message can be sent like a subscription, then it go to the else 
        else:
            await self._removeStream()
            await asyncio.sleep(3600)

    def findSubscription(self, subscription: Subscription) -> Subscription:
        for s in self.subscriptions.keys():
            if s == subscription:
                return s
        return subscription

    async def connectToPeer(self) -> bool:
        ''' Connects to a peer to recieve subscription if it has an active subscription to the stream '''

        async def _isPublisherActive(publisherIp: str) -> bool:
            ''' conirms if the publisher has the subscription stream in its available stream '''
            try:
                response = await self.dataClient.sendRequest(
                    peerHost=publisherIp,
                    uuid=self.streamUuid,
                    method='confirm-subscription'
                )
                if response.status == 'success':
                    return True
                return False
            except Exception as e:
                error('Error connecting to Publisher')
                return False


        if _isPublisherActive(self.publisherHost):
            return True
        else:
            self.peerInfo.subscribersIp = [
                ip for ip in self.peerInfo.subscribersIp if ip != self.serverIp
            ]
            self.rng.shuffle(self.peerInfo.subscribersIp)
            for subscriberIp in self.peerInfo.subcribersIp:
                if _isPublisherActive(subscriberIp):
                    self.publisherHost = subscriberIp
                    return True
        return False
    
    # async def tryToConnect(self):
    #     while not self.isConnectedToPeer:
    #         self.isConnectedToPeer = await self.connectToPeer()
    #         if not self.isConnectedToPeer:
    #             self._removeStream()
    #             time.sleep(600)
        
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
                uuid=self.streamUuid,
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
                    uuid=self.streamUuid,
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
              uuid=self.streamUuid,
              publicationUuid=self.predictionStreamUuid,
              callback=self.handleSubscriptionMessage)

    # def listenToSubscription():
    #     '''
    #     some messages will be on our response variable (subscription.uuid = raw data stream uuid)
    #      - append to the data (handleNewObservation)
    #      - produce prediction if we can (producePrediction)
    #      - pass prediction to our server (create function or add call to server to end of producePrediction)
    #     other message will be on features (subscription.uuid != raw data stream uuid)
    #      - append to the data

    #     callbacks could just start a thread to do these things.
    #     '''
    #     pass

    async def handleSubscriptionMessage(self, subscription: Subscription, message: Message):
        if message.status != 'inactive':
            # we pass the observation to server here instead of inside dataclient?
            self.appendNewData(message.data) # TODO : refactor after confirming sendSubscription Endpoint
            forecast = await self.producePrediction() 
            await self.passPredictionData(forecast) # pass new data and prediction to the server
        else:
            # tell the dataClient to remove the corresponding prediction stream from it's list of publications
            self.isConnectedToPeer = False

            # try to connect to another peer
            # maybe a reconnect function
            await self.init2() # something like this


    async def _addStream(self):
        ''' adds the subscription and publication streams to server avaiable streams '''
        try:
            await self.dataClient.sendRequest(
                    peerHost=self.serverIp,
                    uuid=self.streamUuid,
                    method="add-available-subscription-streams"
                )
        except Exception as e:
            error("Not able to send subscription stream to server")
        try:
            await self.dataClient.sendRequest(
                    peerHost=self.serverIp,
                    uuid=self.predictionStreamUuid,
                    method="add-available-publication-streams"
                )
        except Exception as e:
            error("Not able to send publication stream to server")
        
    async def _removeStream(self, publisherIp):
        ''' removes the subscription and publication streams from server available streams '''
        try:
            await self.dataClient.sendRequest(
                    peerHost=self.serverIp,
                    uuid=self.streamUuid,
                    method="remove-available-subscription-streams"
                )
        except Exception as e:
            error("Not able to send subscription stream to server")
        try:
            await self.dataClient.sendRequest(
                    peerHost=self.serverIp,
                    uuid=self.predictionStreamUuid,
                    method="remove-available-publication-streams"
                )
        except Exception as e:
            error("Not able to send publication stream to server")

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def appendNewData(self, observation: Observation):
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
    def producePrediction(self, updatedModel=None) -> pd.DataFrame:
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
                    return pd.DataFrame({
                        'date_time': [datetimeToTimestamp(now())],
                        'value': [StreamForecast.firstPredictionOf(forecast)],
                        'id': ['random'] # TODO : new hashing method
                    })
                else:
                    raise Exception("Forecast not in dataframe format")
        except Exception as e:
            error(e)
            self.fallback_prediction()

    async def passPredictionData(self, forecast: pd.DataFrame):
        try:
            # send prediction data
            await self.dataClient.passDataToServer(
                peerHost=self.serverIp,
                uuid=self.predictionStreamUuid,
                data=forecast
            )
            # send updated data
            # await self.dataClient.passDataToServer(
            #     peerHost=self.serverIp,
            #     uuid=self.streamUuid,
            #     isSub=True,
            #     data=self.data
            # )
        except Exception as e:
            error('Failed to send Prediction')

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
                    uuid=self.streamUuid,
                    method="stream-data"
                    )
            df = pd.read_json(datasetJson.data, orient='split')
            output_path = os.path.join('csvs', f'{self.streamUuid}.csv')
            df.to_csv(output_path, index=False)
            return df
        except Exception:
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def prediction_data_path(self) -> str:
        return (
            '/Satori/Neuron/data/testprediction/'
            f'{self.predictionStreamUuid}.csv')

    def modelPath(self) -> str:
        return (
            '/Satori/Neuron/models/veda/'
            f'{self.streamUuid}/'
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
                f'AI Engine: stream id {self.streamUuid} '
                f'switching from {self.adapter.__name__} '
                f'to {adapter.__name__} on {self.streamUuid}',
                color='teal')
            self.adapter = adapter
            self.pilot = adapter(uid=self.streamUuid)
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
                                self.streamUuid,
                                print=True)
                            self.producePrediction(self.stable)
                else:
                    debug(f'model training failed on {self.streamUuid} waiting 10 minutes to retry')
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



# NC : gets the peerinfo from the R-server and gives it to its own DS.

# DS : saves that PeerInfo

# EC : asks for the peerInfo (pubSubMap) from (our own) DS
    
# EC : recieves the PeerInfo and then divides sub and pub

# EC : creates thread for each pub-Subscription
    
# EC1 : connectToPeer(), first tries to connect with PublisherPeer (an external peer) information supplied by the r-server

#     if that succeeds then its good ( PublisherPeer )
#     else : tries to connect to other subscribers 

# EC1 : if we're unable to find any connection that has the stream available for subscription - just die, or maybe retry in an hour

# EC1 : else, connection found and then we successfully subscribed to data

# EC1 : let the DS know that they are successfully subscribed (add this stream to availableStream)

# EC1 : let the DS know that they are will publish our prediction datastream (add this stream to availableStream)

# EC1 : recieves a msg or a disconnect event that the publisher is no longer providing
#       - remove the stream from list of active raw data streams/subscriptions/publictions # datastream from the activation list
#       - (when the DS removes it from the list, it also tells all subscribers that it's no longer available, perpetuating the message)
#       - remove the stream from list of active predictve streams/
#       - (when the DS removes it from the list, it also tells all subscribers that it's no longer available, perpetuating the message)

# PublisherPeer: if any problem arises, sends a message to all of its connected servers, ( server removes the peer from its list ).





# subscription data flow

# N: grabs data from the web, sends it to NDC
# NDC: sends it to it's own DS
# DS: saves it to disk, sends it to any subscribing peer (EDC)
# EDC: sends it to it's own DS, triggers callback for all 'subscribing' streamModels in the engine
# SM: saves to ram, triggers a prediction, which it sends to EDC
# EDC: sends prediction to it's own DS
# DS: saves it to disk, sends it to any subscribing peer (EDC) as ancillary data for models


