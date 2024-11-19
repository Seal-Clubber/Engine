import pandas as pd
from typing import Union, Optional, Any
import joblib
import os

from satoriengine.framework.pipelines.interface import TrainingResult

from xgboost import XGBRegressor, XGBClassifier

class XgbPipeline:

    def __init__(self, **kwargs):
        self.model: XGBRegressor = XGBRegressor(eval_metric='mae')

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        try:
            os.makedirs(os.path.dirname(modelpath), exist_ok=True)
            joblib.dump(self.model, modelpath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def fit(self, **kwargs) -> TrainingResult:
        
        pass

    def compare(self, stable: Optional[Any] = None, **kwargs) -> bool:
        """
        Compare stable (model) and pilot models based on their backtest error.

        Args:
            stable: The current stable model
            replace: Whether to replace stable with pilot if pilot performs better

        Returns:
            bool: True if pilot should replace stable, False otherwise
        """
        pass

    def score(self, **kwargs) -> float:
        """
        will score the model.
        """
        pass

    def predict(self, **kwargs) -> Union[None, pd.DataFrame]:
        """
        Make predictions using the stable model

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            Optional[pd.DataFrame]: Predictions if successful, None otherwise
        """
        pass
