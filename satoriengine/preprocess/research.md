basic idea:
xgb approach good for no trend no cycle, if there is we have to extract them
first then add them back in later.

1. no trend - low p below .05, and a negative value is stationary:
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from numpy import log
series = read_csv('international-airline-passengers.csv', header=0, index_col=0, squeeze=True)
X = series.values
X = log(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))

2. no cycle

cycle types:
a. single cycle,
b. multiple cycles,
c. additive or multiplicitive
