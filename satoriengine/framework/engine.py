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
from satoriengine.framework.determine_feature_set import determine_feature_set
from satoriengine.framework.model_creation import model_create_train_test_and_predict
from satoriengine.framework.process import process_data

streamId = StreamId(source='test', stream='test', target='test', author='test')
path = f'./data/{generatePathId(streamId=streamId)}/aggregate.csv'
processedData = process_data(filename=path)
features = determine_feature_set(processedData)
model = model_create_train_test_and_predict(features)
# df = CSVManager.read(filePath=path)
