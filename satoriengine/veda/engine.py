from satoriengine.veda.adapters import ModelAdapter, SKAdapter, StarterAdapter, XgbAdapter, XgbChronosAdapter
from satoriengine.veda.data import StreamForecast, validate_single_entry
from satorilib.logging import INFO, setup, debug, info, warning, error
from satorilib.utils.system import getProcessorCount
from satorilib.utils.time import datetimeToTimestamp, now
from satorilib.datamanager import DataClient, DataServerApi, PeerInfo, Message, Subscription
from satorilib.wallet.evrmore.identity import EvrmoreIdentity 
from satorineuron import config
import asyncio
import pandas as pd
import numpy as np
import threading
import json
import copy
import time
import os
from io import StringIO
from typing import Union
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
        self.streamModels: dict[str, StreamModel] = {}
        self.subscriptions: dict[str, PeerInfo] = {}
        self.publications: dict[str, PeerInfo] = {}
        self.dataServerIp: str = ''
        self.dataClient: Union[DataClient, None] = None
        self.paused: bool = False
        self.threads: list[threading.Thread] = []

    async def initialize(self):
        await self.connectToDataServer()
        asyncio.create_task(self.stayConnectedForever())
        await self.startService()

    async def startService(self): 
        await self.getPubSubInfo()
        await self.initializeModels()
        # await asyncio.Event().wait()

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

    @property
    def isConnectedToServer(self):
        if hasattr(self, 'dataClient') and self.dataClient is not None:
            return self.dataClient.isConnected()
        return False
    
    async def connectToDataServer(self):
        ''' connect to server, retry if failed '''

        async def authenticate() -> bool:
            response = await self.dataClient.authenticate()
            if response.status == DataServerApi.statusSuccess.value:
                return True
            return False

        async def initiateServerConnection() -> bool:
            ''' local engine client authorization '''
            self.dataClient = DataClient(self.dataServerIp, identity=EvrmoreIdentity(config.walletPath('wallet.yaml'))) 
            if await authenticate():
                response = await self.dataClient.isLocalEngineClient()
                if response.status == DataServerApi.statusSuccess.value:
                    info("Local Engine successfully connected to Server Ip at :", self.dataServerIp, color="green")
                    return True
                # raise Exception(response.senderMsg)
            return False
        
        waitingPeriod = 10
        
        while not self.isConnectedToServer:
            try:
                self.dataServerIp = config.get().get('server ip', '0.0.0.0')
                if await initiateServerConnection():
                    return True
            except Exception as e:
                warning(f'Failed to find a valid Server Ip, retrying in {waitingPeriod}')
                await asyncio.sleep(waitingPeriod)

    async def getPubSubInfo(self):
        ''' gets the relation info between pub-sub streams '''
        waitingPeriod = 10
        while not self.subscriptions and self.isConnectedToServer:
            try:
                pubSubResponse: Message = await self.dataClient.getPubsubMap()
                if pubSubResponse.status == DataServerApi.statusSuccess.value and pubSubResponse.streamInfo:
                    for sub_uuid, data in pubSubResponse.streamInfo.items():
                        # TODO : deal with supportive streams, ( data['supportiveUuid'] )
                        self.subscriptions[sub_uuid] = PeerInfo(data['dataStreamSubscribers'], data['dataStreamPublishers'])
                        self.publications[data['publicationUuid']] = PeerInfo(data['predictiveStreamSubscribers'], data['predictiveStreamPublishers'])
                    if self.subscriptions:
                        info(pubSubResponse.senderMsg, color='green')
                else:
                    raise Exception
            except Exception:
                warning(f"Failed to fetch pub-sub info, waiting for {waitingPeriod} seconds")
                await asyncio.sleep(waitingPeriod)

    async def stayConnectedForever(self):
        ''' alternative to await asyncio.Event().wait() '''
        while True:
            await asyncio.sleep(10)
            if not self.isConnectedToServer:
                await self.connectToDataServer()
                await self.startService()
    
    async def initializeModels(self):
        for subUuid, pubUuid in zip(self.subscriptions.keys(), self.publications.keys()): 
            peers = self.subscriptions[subUuid]
            try:
                self.streamModels[subUuid] = await StreamModel.create(
                    streamUuid=subUuid,
                    predictionStreamUuid=pubUuid,
                    peerInfo=peers,
                    dataClient=self.dataClient,
                    pauseAll=self.pause,
                    resumeAll=self.resume)
            except Exception as e:
                error(e)
            debug(3, color='cyan')
            self.streamModels[subUuid].chooseAdapter(inplace=True)
            debug(4, color='cyan')
            self.streamModels[subUuid].run_forever()

    def cleanupThreads(self):
        for thread in self.threads:
            if not thread.is_alive():
                self.threads.remove(thread)
        debug(f'prediction thread count: {len(self.threads)}')


class StreamModel:

    @classmethod
    async def create(
        cls, 
        streamUuid: str,
        predictionStreamUuid: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        pauseAll:callable,
        resumeAll:callable,
    ):
        streamModel = cls(
            streamUuid,
            predictionStreamUuid,
            peerInfo,
            dataClient,
            pauseAll,
            resumeAll,
        )
        await streamModel.initialize()
        return streamModel

    def __init__(
        self,
        streamUuid: str,
        predictionStreamUuid: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        pauseAll:callable,
        resumeAll:callable,
    ):
        self.cpu = getProcessorCount()
        self.pauseAll = pauseAll
        self.resumeAll = resumeAll
        self.preferredAdapters: list[ModelAdapter] = [StarterAdapter, XgbAdapter, XgbChronosAdapter]# SKAdapter #model[0] issue
        self.defaultAdapters: list[ModelAdapter] = [XgbAdapter, XgbAdapter, StarterAdapter]
        self.failedAdapters = []
        self.thread: threading.Thread = None
        self.streamUuid: str = streamUuid
        self.predictionStreamUuid: str = predictionStreamUuid
        self.peerInfo: PeerInfo = peerInfo
        self.dataClient: DataClient = dataClient
        self.rng = np.random.default_rng(37)
        self.publisherHost = self.peerInfo.publishersIp[0]
        self.isConnectedToPublisher = False

    async def initialize(self):
        self.data: pd.DataFrame = await self.loadData()
        self.adapter: ModelAdapter = self.chooseAdapter()
        self.pilot: ModelAdapter = self.adapter(uid=self.streamUuid)
        self.pilot.load(self.modelPath())
        self.stable: ModelAdapter = copy.deepcopy(self.pilot)
        self.paused: bool = False
        debug(f'AI Engine: stream id {self.streamUuid} using {self.adapter.__name__}', color='teal')
        # await self.init2()

    async def init2(self):
        await self.connectToPeer()
        asyncio.create_task(self.stayConnectedToPublisher())
        await self.startStreamService()
    
    async def startStreamService(self):
        await self.syncData()
        await self.makeSubscription()

    @property
    def isConnectedToPublisher(self):
        if hasattr(self, 'dataClient') and self.dataClient is not None:
            return self.dataClient.isConnected(self.publisherHost)
        return False

    async def stayConnectedToPublisher(self):
        while True:
            await asyncio.sleep(10) 
            if not self.isConnectedToPublisher:
                await self.connectToPeer()
                await self.startStreamService()
        

    async def connectToPeer(self) -> bool:
        ''' Connects to a peer to receive subscription if it has an active subscription to the stream '''

        async def _isPublisherActive(publisherIp: str) -> bool:
            ''' conirms if the publisher has the subscription stream in its available stream '''
            try:
                # TODO: authenticate only after confirming the stream is active
                response = await self.dataClient.isStreamActive(publisherIp, self.streamUuid)
                if response.status == DataServerApi.statusSuccess.value:
                    info("successfully connected to an active Publisher Ip at :", publisherIp, color="green")
                    return True
                else:
                    raise Exception(response.senderMsg)
            except Exception as e:
                error('Error connecting to Publisher: ', e)
                return False

        while self.isConnectedToPublisher:
            if await _isPublisherActive(self.publisherHost):
                # auth
                return True
            self.peerInfo.subscribersIp = [
                ip for ip in self.peerInfo.subscribersIp if ip != self.serverIp
            ]
            self.rng.shuffle(self.peerInfo.subscribersIp)
            for subscriberIp in self.peerInfo.subscribersIp:
                if await _isPublisherActive(subscriberIp):
                    # auth
                    self.publisherHost = subscriberIp
                    return True
            await asyncio.sleep(60*60)  
        return False
    
    async def syncData(self):
        '''
        - this can be highly optimized. but for now we do the simple version
        - just ask for their entire dataset every time
            - if it's different than the df we got from our own dataserver,
              then tell dataserver to save this instead
            - replace what we have
        '''
        try:
            externalDataResponse = await self.dataClient.getRemoteStreamData(self.publisherHost, self.streamUuid)
            if externalDataResponse.status == DataServerApi.statusSuccess.value:
                externalDf = pd.read_json(StringIO(externalDataResponse.data), orient='split')
                if not externalDf.equals(self.data) and len(externalDf) > 0: # TODO : sure about this?
                    self.data = externalDf.reset_index().rename(columns={
                        'index': 'date_time',
                        'hash': 'id'
                    })
                    response = await self.dataClient.insertStreamData(
                                    uuid=self.streamUuid,
                                    data=externalDf,
                                    replace=True
                                )
                    if response.status == DataServerApi.statusSuccess.value:
                        info("Data updated in server", color='green')
                    else:
                        raise Exception(externalDataResponse.senderMsg)
            else:
                raise Exception(externalDataResponse.senderMsg)
        except Exception as e:
            error("Failed to sync data, ", e)

    # TODO: after subscribing let others know that we are subscribed to a particular data stream
    async def makeSubscription(self):
        '''
        - and subscribe to the stream so we get the information
            - whenever we get an observation on this stream, pass to the DataServer
        - continually generate predictions for prediction publication streams and pass that to 
        '''
        await self.dataClient.subscribe(
              peerHost=self.publisherHost,
              uuid=self.streamUuid,
              publicationUuid=self.predictionStreamUuid,
              callback=self.handleSubscriptionMessage)

    async def handleSubscriptionMessage(self, subscription: Subscription, message: Message):
        if message.status != DataServerApi.statusInactiveStream:
            self.appendNewData(message.data)
            self.pauseAll()
            await self.producePrediction() 
            self.resumeAll()
        else:
            await self._sendInactive()
            await self.connectToPeer()
            await self.startStreamService()

    async def _sendInactive(self):
        ''' sends stream inactive request to the server so that it can remove the streams from available streams '''
        try:
            response = await self.dataClient.streamInactive(self.streamUuid)
            if response.status != DataServerApi.statusSuccess.value:
                raise Exception(response.senderMsg)
        except Exception as e:
            error("Inactive message not sent to server: ", e)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def appendNewData(self, observation: pd.DataFrame):
        """extract the data and save it to self.data"""
        observationDf = observation.reset_index().rename(columns={
                            'index': 'date_time',
                            'hash': 'id'
                        })
        if validate_single_entry(observationDf['date_time'].values[0], observationDf["value"].values[0]):
            self.data = pd.concat([self.data, observationDf], ignore_index=True)
        else:
            error("Row not added due to corrupt observation")

    
    async def producePrediction(self, updatedModel=None):
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
                    predictionDf = pd.DataFrame({ 'value': [StreamForecast.firstPredictionOf(forecast)]
                                    }, index=[datetimeToTimestamp(now())])
                    await self.passPredictionData(predictionDf) 
                else:
                    raise Exception('Forecast not in dataframe format')
        except Exception as e:
            error(e)
            await self.fallback_prediction()

    async def passPredictionData(self, forecast: pd.DataFrame):
        try:
            response = await self.dataClient.insertStreamData(
                            uuid=self.predictionStreamUuid,
                            data=forecast,
                            isSub=True
                        )
            if response.status == DataServerApi.statusSuccess.value:
                info(response.senderMsg, color='green')
            else:
                raise Exception(response.senderMsg)
        except Exception as e:
            error('Failed to send Prediction to server : ', e)

    async def fallback_prediction(self):
        if os.path.isfile(self.modelPath()):
            try:
                os.remove(self.modelPath())
                debug('Deleted failed model file', color='teal')
            except Exception as e:
                error(f'Failed to delete model file: {str(e)}')
        backupModel = self.defaultAdapters[-1]()
        try:
            trainingResult = backupModel.fit(data=self.data)
            if abs(trainingResult.status) == 1:
                await self.producePrediction(backupModel)
        except Exception as e:
            error(f"Error training new model: {str(e)}")

    async def loadData(self) -> pd.DataFrame:
        try:
            response = await self.dataClient.getLocalStreamData(uuid=self.streamUuid)
            if response.status == DataServerApi.statusSuccess.value:
                return response.data.reset_index().rename(columns={
                    'ts': 'date_time',
                    'hash': 'id'
                })
            else:
                raise Exception(response.senderMsg)
        except Exception as e:
            error(e)
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def modelPath(self) -> str:
        return (
            '/Satori/Neuron/models/veda/'
            f'{self.predictionStreamUuid}/'
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
                            future = asyncio.run_coroutine_threadsafe(
                                self.producePrediction(self.stable),
                                self._loop
                            )
                            future.result()
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
        """
        Creates a new thread for running the model training loop.
        Ensures the thread has access to the event loop for async operations.
        """
        def run_with_loop():

            def run_loop_forever():
                self._loop.run_forever()
                
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            asyncio.set_event_loop(self._loop)
            loop_thread = threading.Thread(target=run_loop_forever, daemon=True)
            loop_thread.start()
            try:
                self.run()
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                loop_thread.join()

        self.thread = threading.Thread(target=run_with_loop, daemon=True)
        self.thread.start()



# this is how we initialize

# async def main():
#     engine = await Engine.create()
#     await asyncio.Event().wait()
#     #await asyncio.Future()
#     #await asyncio.sleep(10)
#     #await asyncio.create_task(client._keepAlive())


    
# asyncio.run(main())
