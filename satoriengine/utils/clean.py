import re


def columnNames(columns: list[str]) -> list[str]:
    '''
    to fix 
    Exception in repeatWrapper: feature_names must be string, and may not contain [, ] or <
    '''
    return [re.sub(r'[^a-zA-Z0-9 _\-#]', '', str(col)) for col in columns]


# Test case
# def test_columnNames():
#    # Create a sample DataFrame with problematic column names
#    import pandas as pd
#    df = pd.DataFrame({
#        'col[1]': [1, 2, 3],
#        'col<2>': [4, 5, 6],
#        'col]3#': [7, 8, 9],
#        'col space': [10, 11, 12],
#        'col#-underscore': [13, 14, 15]
#    })
#    # Apply the columnNames function
#    cleaned_df = columnNames(df)
#    # Expected column names after cleaning
#    expected_columns = ['col1', 'col2', 'col3#', 'col space', 'col#-underscore']
#    # Assertions to check if the cleaned column names match the expected ones
#    assert cleaned_df.columns.tolist() == expected_columns, f"Expected {expected_columns}, but got {cleaned_df.columns.tolist()}"
#
# Run the test
# test_columnNames()
#
# print("All tests passed.")
#
