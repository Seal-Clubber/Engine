from typing import Union
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from satorilib.logging import info, debug
from satoriengine.veda.process import process_data
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult


class XgbPipeline(PipelineInterface):

    def __init__(self, *args, **kwargs):
        self.model: XGBRegressor = None
        self.hyperparameters: Union[dict, None] = None
        self.train_x: pd.DataFrame = None
        self.test_x: pd.DataFrame = None
        self.train_y: np.ndarray = None
        self.test_y: np.ndarray = None
        self.X_full: pd.DataFrame = None
        self.y_full: pd.Series = None
        self.split: float = None
        self.model_error: float = None

    def load(self, modelPath: str, *args, **kwargs) -> Union[None, XGBRegressor]:
        """loads the model model from disk if present"""
        try:
            saved_state = joblib.load(modelPath)
            self.model = saved_state['stable_model']
            self.model_error = saved_state['model_error']
            return self.model
        except Exception as e:
            debug(f"Error Loading Model File : {e}", print=True)
            if os.path.isfile(modelPath):
                os.remove(modelPath)
            return None

    def save(self, modelpath: str, *args, **kwargs) -> bool:
        """saves the stable model to disk"""
        try:
            os.makedirs(os.path.dirname(modelpath), exist_ok=True)
            self.model_error = self.score()
            state = {
                'stable_model' : self.model,
                'model_error' : self.model_error}
            joblib.dump(state, modelpath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def compare(self, other: Union[PipelineInterface, None] = None, *args, **kwargs) -> bool:
        """
        Compare other (model) and this models based on their backtest error.
        Returns True if this model performs better than other model.
        """
        if not isinstance(other, self.__class__):
            return True
        this_score = self.score()
        other_score = other.model_error or other.score()
        is_improved = this_score < other_score
        if is_improved:
            info(
                'model improved!'
                f'\n  stable score: {other_score}'
                f'\n  pilot  score: {this_score}'
                f'\n  Parameters: {self.hyperparameters}',
                color='green')
        else:
            debug(
                f'\nstable score: {other_score}'
                f'\npilot  score: {this_score}',
                color='yellow')
        return is_improved

    def score(self, *args, **kwargs) -> float:
        """will score the model"""
        if self.model is None:
            return np.inf
        self.model_error = mean_absolute_error(self.test_y, self.model.predict(self.test_x))
        return self.model_error

    def fit(self, data: pd.DataFrame, *args, **kwargs) -> TrainingResult:
        """ Train a new model """
        proc_data = process_data(data, quick_start=False)
        # todo: get ready to combine features from different sources (merge)
        # todo: keep a running dataset and update incrementally w/ process_data
        # todo: linear, if not fractal, interpolation
        print('data after', t, proc_data.dataset)
        pre_train_x, pre_test_x, self.train_y, self.test_y = train_test_split(
            proc_data.dataset.index.values,
            proc_data.dataset['value'],
            test_size=self.split or 0.2,
            shuffle=False,
            random_state=37)
        self.train_x = self._prepare_time_features(pre_train_x)
        self.test_x = self._prepare_time_features(pre_test_x)
        self.hyperparameters = self._mutate_params()
        if self.model is None:
            self.model = XGBRegressor(**self.hyperparameters)
        else:
            self.model.set_params(**self.hyperparameters)
        self.model.fit(
            self.train_x,
            self.train_y,
            eval_set=[(self.train_x, self.train_y), (self.test_x, self.test_y)],
            verbose=False)
        return TrainingResult(1, self, False)

    def predict(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """Make predictions using the stable model"""
        proc_data = process_data(data, quick_start=False)
        self.X_full = self._prepare_time_features(proc_data.dataset.index.values)
        self.y_full = proc_data.dataset['value']
        self.model.fit(
            self.X_full,
            self.y_full,
            verbose=False)
        last_date = pd.Timestamp(proc_data.dataset.index[-1])
        future_predictions = self._predict_future(
            self.model,
            last_date,
            proc_data.sampling_frequency)
        return future_predictions

    def _predict_future(
        self,
        model: XGBRegressor,
        last_date: pd.Timestamp,
        sf: str = 'H',
        periods: int = 168,
    ) -> pd.DataFrame:
        """Generate predictions for future periods"""
        future_dates = pd.date_range(
            start=pd.Timestamp(last_date) + pd.Timedelta(sf),
            periods=periods,
            freq=sf)
        future_features = self._prepare_time_features(future_dates)
        predictions = model.predict(future_features)
        results = pd.DataFrame({'date_time': future_dates, 'pred': predictions})
        return results

    @staticmethod
    def _prepare_time_features(dates: np.ndarray) -> pd.DataFrame:
        """Convert datetime series into numeric features for XGBoost"""
        df = pd.DataFrame({'date_time': pd.to_datetime(dates)})
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year
        df['day_of_week'] = df['date_time'].dt.dayofweek
        return df.drop('date_time', axis=1)

    @staticmethod
    def param_bounds() -> dict:
        return {
            'n_estimators': (100, 2000),
            'max_depth': (3, 10),
            'learning_rate': (0.005, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0, 1),
            'scale_pos_weight': (0.5, 10)}

    @staticmethod
    def _prep_params() -> dict:
        """
        Generates randomized hyperparameters for XGBoost within reasonable ranges.
        Returns a dictionary of hyperparameters.
        """
        param_bounds: dict = XgbPipeline.param_bounds()
        rng = np.random.default_rng(37)
        params = {
            'random_state': rng.integers(0, 10000),
            'eval_metric': 'mae',
            'learning_rate': rng.uniform(
                param_bounds['learning_rate'][0],
                param_bounds['learning_rate'][1]),
            'subsample': rng.uniform(
                param_bounds['subsample'][0],
                param_bounds['subsample'][1]),
            'colsample_bytree': rng.uniform(
                param_bounds['colsample_bytree'][0],
                param_bounds['colsample_bytree'][1]),
            'gamma': rng.uniform(
                param_bounds['gamma'][0],
                param_bounds['gamma'][1]),
            'n_estimators': rng.integers(
                param_bounds['n_estimators'][0],
                param_bounds['n_estimators'][1]),
            'max_depth': rng.integers(
                param_bounds['max_depth'][0],
                param_bounds['max_depth'][1]),
            'min_child_weight': rng.integers(
                param_bounds['min_child_weight'][0],
                param_bounds['min_child_weight'][1]),
            'scale_pos_weight': rng.uniform(
                param_bounds['scale_pos_weight'][0],
                param_bounds['scale_pos_weight'][1])}
        return params

    @staticmethod
    def _mutate_params(prev_params: Union[dict, None] = None) -> dict:
        """
        Tweaks the previous hyperparameters for XGBoost by making random adjustments
        based on a squished normal distribution that respects both boundaries and the
        relative position of the current value within the range.
        Args:
            prev_params (dict): A dictionary of previous hyperparameters.
        Returns:
            dict: A dictionary of tweaked hyperparameters.
        """
        prev_params = prev_params or XgbPipeline._prep_params()
        param_bounds: dict = XgbPipeline.param_bounds()
        rng = np.random.default_rng(37)
        mutated_params = {}
        for param, (min_bound, max_bound) in param_bounds.items():
            current_value = prev_params[param]
            range_span = max_bound - min_bound
            # Generate a symmetric tweak centered on the current value
            std_dev = range_span * 0.1  # 10% of the range as standard deviation
            tweak = rng.normal(0, std_dev)
            # Adjust the parameter and ensure it stays within bounds
            new_value = current_value + tweak
            new_value = max(min_bound, min(max_bound, new_value))
            # Ensure integers for appropriate parameters
            if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                new_value = int(round(new_value))
            mutated_params[param] = new_value
        # to handle static parameters... we should keep random_state static
        # because we're exploring the hyperparameter state space relative to it
        mutated_params['random_state'] = prev_params['random_state']
        mutated_params['eval_metric'] = 'mae'
        return mutated_params


    @staticmethod
    def _straight_line_interpolation(df, value_col, step='10T', scale=0.0):
        """
        This would probably be better to use than the stepwise pattern as it
        atleast points in the direction of the trend.
        Performs straight line interpolation on missing timestamps.
        Parameters:
        - df: DataFrame with a datetime index and a column to interpolate.
        - value_col: The column name with values to interpolate.
        - step: The frequency to use for resampling (e.g., '10T' for 10 minutes).
        Returns:
        - DataFrame with interpolated values.
        """
        # Ensure the DataFrame has a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
                df.set_index('date_time', inplace=True)
            else:
                raise ValueError("The DataFrame must have a DatetimeIndex or a 'date_time' column.")
        # Sort the index and resample
        df = df.sort_index()
        df = df.resample(step).mean()  # Resample to fill in missing timestamps with NaN
        # Perform fractal interpolation
        for _ in range(5):  # Number of fractal iterations
            filled = df[value_col].interpolate(method='linear')  # Linear interpolation
            rng = np.random.default_rng(seed=37)
            perturbation = rng.normal(scale=scale, size=len(filled))  # Small random noise
            df[value_col] = filled + perturbation  # Add fractal-like noise
        return df

    @staticmethod
    def merge(dfs: list[pd.DataFrame], targetColumn: Union[str, tuple[str]]):
        ''' Layer 1
        combines multiple mutlicolumned dataframes.
        to support disparate frequencies,
        outter join fills in missing values with previous value.
        filters down to the target column observations.
        '''
        from functools import reduce
        import pandas as pd
        if len(dfs) == 0:
            return None
        if len(dfs) == 1:
            return dfs[0]
        for ix, item in enumerate(dfs):
            if targetColumn in item.columns:
                dfs.insert(0, dfs.pop(ix))
                break
            # if we get through this loop without hitting the if
            # we could possibly use that as a trigger to use the
            # other merge function, also if targetColumn is None
            # why would we make a dataset without target though?
        for df in dfs:
            df.index = pd.to_datetime(df.index)
        return reduce(
            lambda left, right:
                pd.merge_asof(left, right, left_index=True, right_index=True),
            dfs)
