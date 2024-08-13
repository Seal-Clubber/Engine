import pandas as pd


def coerceAndFill(df):
    def coerceColumn(col):
        # Attempt to convert to float
        col = pd.to_numeric(col, errors='coerce')
        # Case 1: If all values are NaN, replace them all with 0
        if col.isna().all():
            return col.fillna(0)
        # Case 2: Not all values are NaN, replace NaNs with the previous value
        col = col.fillna(method='ffill')
        # If the first value is still NaN (because it's the first row), fill it with 0
        col = col.fillna(0)
        return col

    # Apply the function to each column in the DataFrame
    return df.apply(coerceColumn)

# Example usage with your DataFrame
# import numpy as np
# df = pd.DataFrame({
#    'col1': ['1.0', '2.0', 'NaN', '4.0', 'five'],
#    'col2': ['a', 'b', 'c', 'd', 'e'],
#    'col3': ['1', 'two', 'three', np.nan, '4']
# })
#
# Apply the coercion and filling function
# df = coerceAndFill(df)
#
# print(df)
#
