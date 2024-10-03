# AIEngine
#   for each model
#       get it's own data
#       get train and retrain model
#           quickstart (first stable)
#           heavy model (pilot)
#       provide predictions on demand

from satorilib.api.disk.filetypes.csv import CSVManager
from satorilib.api.hash import generatePathId
from satorilib.concepts import StreamId
streamId = StreamId(source='test', stream='test', target='test', author='test')
path = f'./data/{generatePathId(streamId=streamId)}/aggregate.csv'
# df = CSVManager.read(filePath=path)
