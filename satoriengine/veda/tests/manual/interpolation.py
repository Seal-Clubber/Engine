import pandas as pd
from satoriengine.veda.interpolation.rows import add_missing_rows
from satoriengine.veda.interpolation.lomb_scargle import fillMissingValues
from satoriengine.veda.interpolation.lomb_scargle import visualizeToPng
df = pd.read_csv('data', names=['date_time', 'value', 'id'], header=None)
df["date_time"] = pd.to_datetime(df["date_time"])
df.index = df["date_time"]
seconds = df.index.to_series().diff().median().seconds
filled = add_missing_rows(df, seconds=seconds, ffill=False)
visualizeToPng(filled.index, filled['value'], annotations=None, output_path='./viz/original_multiple.png')
interpolated = fillMissingValues(filled)
visualizeToPng(interpolated.index, interpolated['value'], annotations=None, output_path='./viz/combined_mutiple.png')
