import pandas as pd
from datetime import timedelta

def add_missing_rows(
    dataframe: pd.DataFrame,
    seconds: int,
    ffill:bool = True,
) -> pd.DataFrame:
    """
    Adds rows to a DataFrame where the time difference between consecutive rows
    exceeds twice the given number of seconds. The new rows are added at the expected intervals.
    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        seconds (int): Number of seconds to define the expected interval.
    Returns:
        pd.DataFrame: Updated DataFrame with missing rows filled.
    """
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    # Calculate the expected frequency as a timedelta
    freq_timedelta = timedelta(seconds=seconds)
    # Initialize the new rows list
    new_rows = []
    # Iterate over the DataFrame to detect gaps
    prev_index = dataframe.index[0]
    for current_index in dataframe.index[1:]:
        time_diff = current_index - prev_index
        # If the time difference exceeds twice the expected frequency
        while time_diff > 2 * freq_timedelta:
            # Calculate the next expected timestamp
            prev_index += freq_timedelta
            # Add a new row with NaN values
            new_row = {col: pd.NA for col in dataframe.columns}
            new_rows.append((prev_index, new_row))
            # Update the time difference
            time_diff = current_index - prev_index
        prev_index = current_index
    # Create a DataFrame for the new rows
    new_rows_df = pd.DataFrame(
        [row for index, row in new_rows],
        index=[index for index, row in new_rows]
    )
    # Combine the original DataFrame with the new rows and sort by index
    updated_dataframe = pd.concat([dataframe, new_rows_df]).sort_index()
    if ffill:
        return updated_dataframe.fillna(method='ffill')
    return updated_dataframe


def combine_dataframes(df1, df2):
    """
    Combines two DataFrames with datetime indexes.
    Dynamically detects the highest _i index in df1 and renames all df2 columns with _i+1.
    Args:
        df1, df2: DataFrames to combine, both with datetime indexes.
    Returns:
        Combined DataFrame with merged columns, preserving all data.
    """
    # Find the highest _i index in df1's columns
    def get_max_index(columns):
        max_index = 0
        for col in columns:
            if "_" in col and col.split("_")[-1].isdigit():
                max_index = max(max_index, int(col.split("_")[-1]))
        return max_index
    # Get the next index for df2
    next_index = get_max_index(df1.columns) + 1
    # Rename all columns in df2 with _next_index
    new_columns = {col: f"{col}_{next_index}" for col in df2.columns}
    df2 = df2.rename(columns=new_columns)
    # Combine indexes to ensure no rows are lost
    combined_index = df1.index.union(df2.index).sort_values()
    # Reindex both DataFrames to the combined index, forward-filling missing rows
    df1 = df1.reindex(combined_index).ffill()
    df2 = df2.reindex(combined_index).ffill()
    # Concatenate DataFrames along columns
    combined_df = pd.concat([df1, df2], axis=1)
    return combined_df

if __name__ == "__main__":
    def test_add_missing_rows():
        data = {
            "date_time": [
                "2025-01-04 21:21:10.677600",
                "2025-01-04 21:41:22.751513",
                "2025-01-04 22:01:10.677600",
                "2025-01-04 23:11:11.683042"],
            "value": [6.180800, 6.180800, 6.180800, 6.180800],
            "id": ["eb502gsefesf938125", "eb5022f7ec938125", "acc56096d8f75f85", "58bc522fb6425c40"]
        }
        df = pd.DataFrame(data)
        df["date_time"] = pd.to_datetime(df["date_time"])
        df.set_index("date_time", inplace=True)
        # Call the function
        filled_df = add_missing_rows(df, seconds=60*10)
        print(filled_df)

    def test_combine_dataframes():
        # Example DataFrames
        data1 = {
            "date_time": [
                "2025-01-04 21:21:10.677600",
                "2025-01-04 21:41:22.751513",
                "2025-01-04 22:01:10.677600",
                "2025-01-04 23:11:11.683042"],
            "value": [1, 2, 3, 4],
            "other": [3, 1, 5, 6],
            "value_2": [5, 6, 7, 8],
        }
        data2 = {
            "date_time": [
                "2025-01-04 21:31:15.123456",
                "2025-01-04 21:51:20.654321",
                "2025-01-04 22:11:00.111111"],
            "value": [10, 20, 30],
            "other": [78, 32, 18]
        }
        # Create DataFrames
        df1 = pd.DataFrame(data1).set_index(pd.to_datetime(data1['date_time'])).drop(columns='date_time')
        df2 = pd.DataFrame(data2).set_index(pd.to_datetime(data2['date_time'])).drop(columns='date_time')
        # Combine DataFrames
        result = combine_dataframes(df1, df2)
        print(result)

    test_add_missing_rows()
    test_combine_dataframes()
