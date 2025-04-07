from satoriengine.veda.adapters.multivariate.mvadapters import FastMVAdapter, LightMVAdapter, HeavyMVAdapter
import pandas as pd
import datetime
import os
import joblib

print("reading csvs")
targetDf = pd.read_csv('datasets/co_created/dataset_main.csv', names=['date_time', 'value', 'id'], header=None)
covDf1 = pd.read_csv('datasets/co_created/dataset_correlated.csv', names=['date_time', 'value', 'id'], header=None)
covDf2 = pd.read_csv('datasets/co_created/dataset_non_correlated.csv', names=['date_time', 'value', 'id'], header=None)
# covDf2 = pd.read_csv('datasets/co/3.csv', names=['date_time', 'value', 'id'], header=None)
# covDf1 = pd.read_csv('datasets/co/2.csv', names=['date_time', 'value', 'id'], header=None)
# covDf3 = pd.read_csv('datasets/aggregate.csv', names=['date_time', 'value', 'id'], header=None)
# covDf4 = pd.read_csv('datasets/test.csv', names=['date_time', 'value', 'id'], header=None)

# adap = FastMVAdapter()
# adap = LightMVAdapter()
adap = HeavyMVAdapter()

print("start fitting")
current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")

model = adap.fit(targetDf[:-4], [covDf1[:-4], covDf2[:-4]])

current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
print("done fitting")

os.makedirs("models", exist_ok=True)
state = {'stableModel': model}
joblib.dump(state, 'models/heavy1.joblib')

model = joblib.load('models/heavy1.joblib')['stableModel'].model

current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
print("Predicting")

resultDf = model.predict(targetDf[:-1], [covDf1[:-1], covDf2[:-1]])
# resultDf = model.model.predict(targetDf, [covDf1])

current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
print(resultDf)