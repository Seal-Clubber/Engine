from satoriengine.veda.adapters.multivariate.mvadapters import FastMVAdapter, LightMVAdapter, HeavyMVAdapter
import pandas as pd


vpsAda = LightMVAdapter()
fastAda= FastMVAdapter()
# targetDf = pd.read_csv('datasets/500.csv', names=['date_time', 'value', 'id'], header=None)
targetDf = pd.read_csv('datasets/test.csv', names=['date_time', 'value', 'id'], header=None)
covDf1 = pd.read_csv('datasets/10-2.csv', names=['date_time', 'value', 'id'], header=None)
covDf2 = pd.read_csv('datasets/100.csv', names=['date_time', 'value', 'id'], header=None)
covDf3 = pd.read_csv('datasets/aggregate.csv', names=['date_time', 'value', 'id'], header=None)
# covDf4 = pd.read_csv('datasets/test.csv', names=['date_time', 'value', 'id'], header=None)
# model = vpsAda.fit(targetDf, [covDf1])
model = vpsAda.fit(targetDf, [covDf1, covDf2, covDf3])
print("done fitting")
resultDf = vpsAda.predict(targetDf, [covDf1, covDf2, covDf3])
print(resultDf)