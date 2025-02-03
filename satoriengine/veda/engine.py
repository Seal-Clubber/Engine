from satoriengine.veda.adapters import ModelAdapter, SKAdapter, StarterAdapter, XgbAdapter, XgbChronosAdapter
from satoriengine.veda.data import StreamForecast, validate_single_entry
from satorilib.logging import INFO, setup, debug, info, warning, error
from satorilib.utils.system import getProcessorCount
from satorilib.utils.time import datetimeToTimestamp, now
from Lib.satorilib.datamanager import DataClient, DataServerApi, PeerInfo, Message, Subscription
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
        self.subcriptions: dict[str, PeerInfo] = {}
        self.publications: dict[str, PeerInfo] = {}
        self.dataServerIp: str = ''
        self.dataClient: Union[DataClient, None] = None,
        self.isConnectedToServer: bool = False
        self.paused: bool = False
        self.threads: list[threading.Thread] = []

    async def initialize(self):
        await self.connectToDataServer()
        await self.getPubSubInfo()
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
        ''' connect to server, retry if failed '''

        async def initiateServerConnection() -> bool:
            ''' local engine client authorization '''

            self.dataClient = DataClient(self.dataServerIp)
            response = await self.dataClient.isLocalEngineClient()
            if response.status == DataServerApi.statusSuccess.value:
                info("Local Engine successfully connected to Server Ip at :", self.dataServerIp, color="green")
                return True
            raise Exception(response.senderMsg)
        
        while not self.isConnectedToServer:
            try:
                self.dataServerIp = config.get().get('server ip', '0.0.0.0')
                if await initiateServerConnection():
                    self.isConnectedToServer = True
            except Exception as e:
                error("Error connecting to server ip in config : ", e)
                try:
                    self.dataServerIp = self.start.server.getPublicIp().text.split()[-1] # TODO : is this correct?
                    if await initiateServerConnection():
                        self.isConnectedToServer = True
                except Exception as e:
                    error("Failed to find a valid Server Ip : ", e)
                    info("Retrying connection in 1 hour...")
                    self.isConnectedToServer = False
                    await asyncio.sleep(60*60)

    async def getPubSubInfo(self):
        ''' gets the relation info between pub-sub streams '''

        try:
            pubSubResponse: Message = await self.dataClient.getPubsubMap()
            if pubSubResponse.status == DataServerApi.statusSuccess.value:
                for sub_uuid, data in pubSubResponse.streamInfo.items():
                    self.subcriptions[sub_uuid] = PeerInfo(data['dataStreamSubscribers'], data['dataStreamPublishers'])
                    self.publications[data['publicationUuid']] = PeerInfo(data['predictiveStreamSubscribers'], data['predictiveStreamPublishers'])
                debug(pubSubResponse.senderMsg, print=True)
            else:
                raise Exception(pubSubResponse.senderMsg)
        except Exception as e:
            error(f"Failed to fetch pub-sub info, {e}")
    
    async def initializeModels(self):
        for subuuid, pubuuid in zip(self.subcriptions.keys(), self.publications.keys()): 
            peers = self.subcriptions[subuuid]
            self.streamModels[subuuid] = await StreamModel.create(
                streamId=subuuid,
                predictionStreamId=pubuuid,
                serverIp=self.dataServerIp,
                peerInfo=peers,
                dataClient=self.dataClient,
                pauseAll=self.pause,
                resumeAll=self.resume)
            self.streamModels[subuuid].chooseAdapter(inplace=True)
            self.streamModels[subuuid].run_forever()

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
        serverIp: str,
        peerInfo: PeerInfo,
        dataClient: DataClient,
        pauseAll:callable,
        resumeAll:callable,
    ):
        streamModel = cls(
            streamUuid,
            predictionStreamUuid,
            serverIp,
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
        serverIp: str,
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
        self.serverIp = serverIp
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
        await self.init2()

    async def init2(self):
        await self.connectToPeer()
        await self.syncData()
        await self.makeSubscription()

    async def connectToPeer(self) -> bool:
        ''' Connects to a peer to receive subscription if it has an active subscription to the stream '''

        async def _isPublisherActive(publisherIp: str) -> bool:
            ''' conirms if the publisher has the subscription stream in its available stream '''
            try:
                response = await self.dataClient.isStreamActive(publisherIp, self.streamUuid)
                if response.status == DataServerApi.statusSuccess.value:
                    info("successfully connected to an active Publisher Ip at :", publisherIp, color="green")
                    return True
                else:
                    raise Exception(response.senderMsg)
            except Exception as e:
                error('Error connecting to Publisher: ', e)
                return False

        while not self.isConnectedToPublisher:  
            if await _isPublisherActive(self.publisherHost):
                self.isConnectedToPublisher = True
                return True
            self.peerInfo.subscribersIp = [
                ip for ip in self.peerInfo.subscribersIp if ip != self.serverIp
            ]
            self.rng.shuffle(self.peerInfo.subscribersIp)
            for subscriberIp in self.peerInfo.subscribersIp:
                if await _isPublisherActive(subscriberIp):
                    self.publisherHost = subscriberIp
                    self.isConnectedToPublisher = True
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
                externalDf = pd.read_json(externalDataResponse.data, orient='split')
                if not externalDf.equals(self.data) and len(externalDf) > 0:
                    self.data = externalDf
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
            forecast = await self.producePrediction() 
            await self.passPredictionData(forecast) 
            self.resumeAll()
        else:
            await self._sendInactive(message)
            self.isConnectedToPublisher = False
            await self.init2() 

    async def _sendInactive(self, message: Message = None):
        ''' sends stream inactive request to the server so that it can remove the streams from available streams '''
        try:
            response = await self.dataClient.streamInactive(
                            uuid=self.streamUuid,
                            # isSub=True # TODO : should we add isSub?
                        )
            if response.status != DataServerApi.statusSuccess.value:
                raise Exception(response.senderMsg)
        except Exception as e:
            error("Inactive message not sent to server: ", e)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    # TODO : refactor after confirming how the subscription message is sent
    def appendNewData(self, observation: json):
        """extract the data and save it to self.data"""
        observationDf = pd.read_json(StringIO(observation), orient='split').reset_index().rename(columns={
                            'index': 'date_time',
                            'hash': 'id'
                        })
        if validate_single_entry(observationDf['date_time'].values[0], observationDf["value"].values[0]):
            self.data = pd.concat([self.data, observationDf], ignore_index=True)
        else:
            error("Row not added due to corrupt observation")

    
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
                        'value': [StreamForecast.firstPredictionOf(forecast)]
                    })
                else:
                    raise Exception('Forecast not in dataframe format')
        except Exception as e:
            error(e)
            self.fallback_prediction()

    async def passPredictionData(self, forecast: pd.DataFrame):
        try:
            response = await self.dataClient.insertStreamData(
                            uuid=self.predictionStreamUuid,
                            data=forecast,
                            isSub=True
                        )
            if response.status == DataServerApi.statusSuccess.value:
                info('Prediction Data saved in Server', color='green')
            else:
                raise Exception(response.senderMsg)
        except Exception as e:
            error('Failed to send Prediction to server : ', e)

    def fallback_prediction(self):
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
                self.producePrediction(backupModel)
        except Exception as e:
            error(f"Error training new model: {str(e)}")

    async def loadData(self) -> pd.DataFrame:
        try:
            response = await self.dataClient.getLocalStreamData(uuid=self.streamUuid)
            if response.status == DataServerApi.statusSuccess.value:
                df = pd.read_json(StringIO(response.data), orient='split').reset_index().rename(columns={
                        'index': 'date_time',
                        'hash': 'id'
                    })
                output_path = os.path.join('csvs', f'{self.streamUuid}.csv')
                df.to_csv(output_path, index=False)
                return df
            else:
                raise Exception(response.senderMsg)
            # TODO : after testing just return the below
            # return pd.read_json(StringIO(datasetJson.data), orient='split').reset_index().rename(columns={
            #         'index': 'date_time',
            #         'hash': 'id'
            #     })
        except Exception:
            return pd.DataFrame(columns=["date_time", "value", "id"])

    # def prediction_data_path(self) -> str:
    #     return (
    #         '/Satori/Neuron/data/testprediction/'
    #         f'{self.predictionStreamUuid}.csv')

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



# TODO : this is how we initialize

# async def main():
#     engine = await Engine.create()
#     await asyncio.Event().wait()
#     #await asyncio.Future()
#     #await asyncio.sleep(10)
#     #await asyncio.create_task(client._keepAlive())


    
# asyncio.run(main())
