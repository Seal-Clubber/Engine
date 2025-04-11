import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame
import holidays
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
)


def createTrainTest(data, forecasting_steps: int = 1, includeMulFeatures: bool = False, isUS: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data['date_time'] = pd.to_datetime(data['date_time'])  #convert to datetime
    timeseriesid = 1
    hour_period = 24
    data["hour"] = data["date_time"].dt.hour
    # covariates["hour_sin"] = sin_transformer(24).fit_transform(covariates)["hour"]
    data["hour_sin"] = np.sin(data["hour"] / hour_period * 2 * np.pi)
    # covariates["hour_cos"] = cos_transformer(24).fit_transform(covariates)["hour"]
    data["hour_cos"] = np.cos(data["hour"] / hour_period * 2 * np.pi)
    # cos_transformer(24).fit_transform(covariates)["hour"]

    dayofweek_period = 7
    data["dayofweek"] =  data["date_time"].dt.dayofweek
    data["dayofweek_sin"] = np.sin(data["dayofweek"] / dayofweek_period * 2 * np.pi)
    data["dayofweek_cos"] = np.cos(data["dayofweek"] / dayofweek_period * 2 * np.pi)

    # We could do the sin and cos transform for month, week, or quarter
    
    # WEEKEND_INDICES = [5, 6] # new
    # data["weekend"] = timestamps.weekday.isin(WEEKEND_INDICES).astype(float)
    # data["month"] = timestamps.month
    # data["week"] = timestamps.week
    # data["quarter"] = timestamps.quarter
    data["dayofyear"] = data["date_time"].dt.dayofyear
    data["year"] = data["date_time"].dt.year

    data['timeseriesid'] = timeseriesid #add new colum 'id' to dataframe

    if includeMulFeatures:
        # Order 2 multiplicative interactions
        # ==============================================================================
        transformer_multiply = PolynomialFeatures(
                            degree           = 2,
                            interaction_only = True,
                            include_bias     = False
                        ).set_output(transform="pandas")
        multiplicative_cols = [
            'value',
            'hour_sin', 
            'hour_cos',
            'dayofweek_sin',
            'dayofweek_cos',
            'dayofyear'
        ]
        multiplicative_features = transformer_multiply.fit_transform(data[multiplicative_cols])
        multiplicative_features = multiplicative_features.drop(columns=multiplicative_cols)

        # multiplicative_features.columns = [f"mult_{col}" for col in multiplicative_features.columns]
        multiplicative_features.columns = [f"{col}" for col in multiplicative_features.columns]
        multiplicative_features.columns = multiplicative_features.columns.str.replace(" ", "*")
        assert all(multiplicative_features.index == data.index)

        data = pd.concat([data, multiplicative_features], axis=1)

    data = TimeSeriesDataFrame.from_data_frame(
        data,
        id_column="timeseriesid",
        timestamp_column="date_time"
    )
    num_rows = data.num_timesteps_per_item()[1]
    timestamps = data.index.get_level_values("timestamp")
    # data["hour"] = timestamps.hour
    # data["dayofweek"] = timestamps.weekday
    # # data["weekend"] = timestamps.weekday.isin(WEEKEND_INDICES).astype(float)
    # # data["month"] = timestamps.month
    # data["dayofyear"] = timestamps.dayofyear   
    # data["year"] = timestamps.year    
    # data["quarter"] = timestamps.quarter
    if isUS:
        US_holidays = holidays.country_holidays(
            country="US",  # make sure to select the correct country/region!
            # Add + 1 year to make sure that holidays are initialized for the forecast horizon
            years=range(timestamps.min().year, timestamps.max().year + 1),
        )
        data = add_holiday_features(data, US_holidays)
    data_train, data_traintest = data.train_test_split(
        prediction_length = forecasting_steps
    )
    covariateColNames = [col for col in data_traintest.columns if col not in ['date_time', 'value']]
    return data_train, data_traintest, covariateColNames

def conformData(target_df: pd.DataFrame, covariate_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    target_df = target_df.copy()
    target_df['date_time'] = pd.to_datetime(target_df['date_time'])
    target_df = target_df.set_index('date_time')
    start_time = target_df.index.min()
    end_time = target_df.index.max()
    sf = getSamplingFreq(target_df)
    full_range = pd.date_range(start=start_time, end=end_time, freq=sf)
    target_df = target_df.reindex(full_range, method='nearest').ffill()
    result_df = pd.DataFrame({'value': target_df['value']})
    result_df = result_df.reset_index().rename(columns={'index': 'date_time'})
    for i, cov_df in enumerate(covariate_dfs, 1):
        cov_df = cov_df.copy()
        cov_df['date_time'] = pd.to_datetime(cov_df['date_time'])
        cov_df = cov_df.set_index('date_time').sort_index()
        aligned_cov = cov_df.reindex(
            pd.DatetimeIndex(sorted(set(cov_df.index) | set(full_range))),
            method='ffill'
        )
        aligned_cov = aligned_cov.reindex(full_range)
        cov_col_name = f'covariate_{i}_value'
        result_df[cov_col_name] = aligned_cov['value'].values
    return result_df

def getSamplingFreq(dataset: pd.DataFrame) -> str:
    def fmt(sf):
        return "".join(
            f"{v}{abbr[k]}"
            for k, v in sf.components._asdict().items()
            if v != 0 and k in ["days", "hours", "minutes"])
        
    sf = dataset.index.to_series().diff().median()
    abbr = {
        "days": "d",
        "hours": "h",
        "minutes": "min",
        "seconds": "s",
        "milliseconds": "ms",
        "microseconds": "us",
        "nanoseconds": "ns"}
    if isinstance(sf, pd.Timedelta):
        sampling_frequency = fmt(sf)
    elif isinstance(sf, pd.TimedeltaIndex):
        sampling_frequency = sf.map(fmt)
    else:
        raise ValueError
    return sampling_frequency        

def add_holiday_features(
    data: TimeSeriesDataFrame,
    country_holidays: dict,
    include_holiday_indicator: bool = True,
) -> TimeSeriesDataFrame:
    """Add holiday indicator column to a TimeSeriesDataFrame."""
    data = data.copy()
    timestamps = data.index.get_level_values("timestamp")
    country_holidays_df = pd.get_dummies(pd.Series(country_holidays)).astype(float)
    holidays_df = country_holidays_df.reindex(timestamps.date).fillna(0)
    if include_holiday_indicator:
        data["holiday"] = holidays_df.max(axis=1).values
    return data