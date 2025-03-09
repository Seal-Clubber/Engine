from satoriengine.veda.adapters.multivariate.mvadapters import FastMVAdapter, LightMVAdapter, HeavyMVAdapter
import pandas as pd
import datetime
import os
import joblib

vpsAda = LightMVAdapter()
fastAda= FastMVAdapter()
# targetDf = pd.read_csv('datasets/500.csv', names=['date_time', 'value', 'id'], header=None)
print("reading csvs")
targetDf = pd.read_csv('datasets/co/1.csv', names=['date_time', 'value', 'id'], header=None)
covDf1 = pd.read_csv('datasets/co/2.csv', names=['date_time', 'value', 'id'], header=None)
covDf2 = pd.read_csv('datasets/co/3.csv', names=['date_time', 'value', 'id'], header=None)
# covDf3 = pd.read_csv('datasets/aggregate.csv', names=['date_time', 'value', 'id'], header=None)
# covDf4 = pd.read_csv('datasets/test.csv', names=['date_time', 'value', 'id'], header=None)
# print(targetDf[:-2].tail(10))
print("start fitting")
current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
model = vpsAda.fit(targetDf[:-2], [covDf1[:-2], covDf2[:-2]])
# model = fastAda.fit(targetDf, [covDf1, covDf2, covDf3])
current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
print("done fitting")
os.makedirs("models", exist_ok=True)
state = {'stableModel': model}
joblib.dump(state, 'models/vpssn.joblib')

model = joblib.load('models/vpssn.joblib')['stableModel'].model
current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
print("Predicting")
resultDf = model.predict(targetDf, [covDf1, covDf2])
current_time = datetime.datetime.now()
print(f"Current time with milliseconds: {current_time.strftime('%H:%M:%S.%f')[:-3]}")
print(resultDf)