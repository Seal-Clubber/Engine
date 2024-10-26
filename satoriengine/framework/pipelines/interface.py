import pandas as pd
from typing import Union, Optional, Any
import joblib


class TrainingResult:

    def __init__(self, status, model: "PipelineInterface", stagnated: bool = False):
        self.status = status
        self.model = model
        self.stagnated = stagnated


class PipelineInterface:
    @staticmethod
    def compare(stable: Optional[Any] = None, pilot: Optional[Any] = None) -> bool:
        """
        Compare stable (model) and pilot models based on their backtest error.

        Args:
            stable: The current stable model
            replace: Whether to replace stable with pilot if pilot performs better

        Returns:
            bool: True if pilot should replace stable, False otherwise
        """
        pass

    @staticmethod
    def predict(**kwargs) -> Union[None, pd.DataFrame]:
        """
        Make predictions using the stable model

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            Optional[pd.DataFrame]: Predictions if successful, None otherwise
        """
        pass

    @staticmethod
    def save(model: Optional[Any], modelpath: str) -> bool:
        """
        Save the model to disk.

        Args:
            model: The model to save
            modelpath: Path where the model should be saved

        Returns:
            bool: True if save successful, False otherwise
        """
        pass

    @staticmethod
    def load(modelPath: str) -> Union[None, Any]:
        """loads the model model from disk if present"""

    @staticmethod
    def train(**kwargs) -> TrainingResult:
        """
        Train a new model.

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            TrainingResult: Object containing training status and model
        """
        pass
