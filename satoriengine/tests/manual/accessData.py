from typing import Union
import os
import pandas as pd
from satorilib import logging
from satorilib.concepts import StreamId
from satorilib.api import memory
from satorilib.api.time import datetimeToTimestamp, now, datetimeToTimestamp, earliestDate
from satorilib.api.hash import hashIt, generatePathId, historyHashes, verifyHashes, cleanHashes, verifyRoot, verifyHashesReturnError
from satorilib.api.disk import Disk
from satorilib.api.disk.utils import safetify, safetifyWithResult
from satorilib.api.disk.model import ModelApi
from satorilib.api.disk.wallet import WalletApi
from satorilib.api.disk.filetypes.csv import CSVManager
from satorilib.api.disk.cache import Cache
c = Cache(
    id=StreamId(source='satori', author='0358f063ce97bc764df0198d1a66188b550fb1d635101d4995e24ca5b8892881fe',
                stream='Coinbase.USD.PAX_p', target='data.rates.PAX'),
    loc=r'/Satori/Neuron/data')
timestamp = datetimeToTimestamp(now())
value = 15.44404125213623
c.df
c.getHashBefore(timestamp)
c.search(timestamp, before=True)
c.loadCache()
