import pandas as pd
from satorilib import logging


def rawDataMetric(df: pd.DataFrame = None, column: tuple = None) -> pd.Series:

    def name() -> tuple:
        return column

    if df is None:
        return name()
    feature = df.loc[:, column]
    feature.name = name()
    return feature


def dailyPercentChangeMetric(
    df: pd.DataFrame = None,
    column: tuple = None,
    prefix: str = 'Daily',
    yesterday: int = 1,
) -> pd.Series:

    def name() -> tuple:
        return tuple([col for col in column[0:len(column)-1]]) + (prefix+column[len(column)-1]+str(yesterday),)

    if df is None:
        return name()
    feature = df.loc[:, column].shift(
        yesterday-1) / df.loc[:, column].shift(yesterday)
    feature.name = name()
    return feature


def rollingPercentChangeMetric(
    df: pd.DataFrame = None,
    column: tuple = None,
    prefix: str = 'Rolling',
    window: int = 2,
    transformation: str = 'max',
) -> pd.Series:

    def name() -> tuple:
        return tuple([col for col in column[0:len(column)-1]]) + (prefix+column[len(column)-1]+str(window)+transformation.replace('(', '').replace(')', ''),)

    if df is None:
        return name()
    transactionOptions = 'sum max min mean median std count var skew kurt quantile cov corr apply'
    if (isinstance(window, int)
            and transformation.startswith(tuple(transactionOptions.split()))):
        df = df.apply(lambda col: pd.to_numeric(col, errors='ignore'))
        try:
            feature = (
                df[column] /
                eval(
                    f'df[column].shift(1).rolling(window={window}).'
                    f'{transformation}'))
        except Exception as e:
            # don't worry too much about feature engineering right now
            logging.error('engine error', e)
            feature = df[column]
            # if e == "unsupported operand type(s) for /: 'str' and 'float'":
            #    feature = (
            #        df[column].astype(float) /
            #        eval(
            #            f'df[column].shift(1).rolling(window={window}).'
            #            f'{transformation}'))
        feature.name = name()
        return feature

    raise Exception(
        'eval call on transformation failed, unable to create feature')
