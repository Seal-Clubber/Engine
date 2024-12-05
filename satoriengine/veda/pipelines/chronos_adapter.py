from typing import Union
import os
import time
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult


class ChronosVedaPipeline(PipelineInterface):
    def __init__(self, useGPU: bool = False, **kwargs):
        hfhome = os.environ.get(
            'HF_HOME', default='/Satori/Neuron/models/huggingface')
        os.makedirs(hfhome, exist_ok=True)
        device_map = 'cuda' if useGPU else 'cpu'
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large" if useGPU else "amazon/chronos-t5-small",
            # "amazon/chronos-t5-tiny", # 8M
            # "amazon/chronos-t5-mini", # 20M
            # "amazon/chronos-t5-small", # 46M
            # "amazon/chronos-t5-base", # 200M
            # "amazon/chronos-t5-large", # 710M
            # 'cpu' for any CPU, 'cuda' for Nvidia GPU, 'mps' for Apple Silicon
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            # force_download=True,
        )
        self.ctx_len = 512  # historical context

    def fit(self, trainX, trainY, eval_set, verbose):
        ''' online learning '''
        return TrainingResult(1, self, False)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        data = data.values  # Convert DataFrame to numpy array
        # Squeeze only if the first dimension is 1
        if len(data.shape) > 1 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        data = data[-self.ctx_len:]  # Use the last `ctx_len` rows
        context = torch.tensor(data)
        t1_start = time.perf_counter_ns()
        forecast = self.model.predict(
            context,
            1,  # prediction_length
            num_samples=4,  # 20
            temperature=1.0,  # 1.0
            top_k=64,  # 50
            top_p=1.0,  # 1.0
        )  # forecast shape: [num_series, num_samples, prediction_length]
        # low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        median = forecast.median(dim=1).values
        predictions = median[0]
        total_time = (time.perf_counter_ns() - t1_start) / 1e9  # seconds
        print(
            f"Chronos prediction time seconds: {total_time}    Historical context size: {data.shape}    Predictions: {predictions}")
        return np.asarray(predictions, dtype=np.float32)

    def compare(self, other: Union[PipelineInterface, None] = None, **kwargs) -> bool:
        """
        Compare other (model) and this models based on their backtest error.
        Returns True if this model performs better than other model.
        """
        return kwargs.get('override', True)

    def score(self, **kwargs) -> float:
        """will score the model"""
        return np.inf


def generate_training_data(ctx_len: int, num_features: int = 1, num_samples: int = 1000):
    """
    Generates synthetic training data for testing.
    Parameters:
    - ctx_len: The historical context length required for each sample (rows).
    - num_features: The number of features (columns) in the training data.
    - num_samples: The number of samples in the training set.
    Returns:
    - trainX: A list of pandas DataFrames, where each DataFrame represents one sample.
    - trainY: A numpy array of shape (num_samples, 1).
    - eval_set: A tuple containing evaluation X and Y datasets.
    """
    import pandas as pd
    import numpy as np
    # Generate synthetic training data
    trainX = []
    trainY = []
    for _ in range(num_samples):
        # Generate a sine wave with some noise
        data = np.sin(np.linspace(0, 20 * np.pi, ctx_len)) + \
            np.random.normal(0, 0.1, ctx_len)
        # Multiple features: stack the same time series with small variations
        features = np.stack([data + np.random.normal(0, 0.01, ctx_len)
                            for _ in range(num_features)], axis=1)
        # Target is the mean of the last 10 points
        target = np.mean(data[-10:]) + np.random.normal(0, 0.1)
        # Convert features to DataFrame
        df = pd.DataFrame(features, columns=[
                          f'feature_{i}' for i in range(num_features)])
        trainX.append(df)
        trainY.append(target)
    trainY = np.array(trainY)
    # Create evaluation set
    eval_size = max(10, int(num_samples * 0.1))
    evalX = trainX[:eval_size]
    evalY = trainY[:eval_size]
    eval_set = (evalX, evalY)
    return trainX, trainY, eval_set


# Generate training data
ctx_len = 512
num_features = 3  # Example for multiple features
trainX, trainY, eval_set = generate_training_data(
    ctx_len=ctx_len, num_features=num_features, num_samples=1000)

# Pass the first sample as a DataFrame to predict
c = ChronosVedaPipeline()
c.predict(trainX[0])
