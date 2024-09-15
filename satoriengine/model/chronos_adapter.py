import time
import numpy as np
import torch
from chronos import ChronosPipeline

class ChronosAdapter():

    def __init__(self, useGPU):
        device_map = 'cuda' if useGPU else 'cpu'
        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny", # 8M
            # "amazon/chronos-t5-mini", # 20M
            # "amazon/chronos-t5-small", # 46M
            # "amazon/chronos-t5-base", # 200M
            # "amazon/chronos-t5-large", # 710M
            device_map=device_map, # 'cpu' for any CPU, 'cuda' for Nvidia GPU, 'mps' for Apple Silicon
            torch_dtype=torch.bfloat16,
            # force_download=True,
        )
        self.ctx_len = 512 # historical context

    def fit(self, trainX, trainY, eval_set, verbose):
        pass

    def predict(self, current):
        data = current.values
        data = np.squeeze(data)
        data = data[-self.ctx_len:] # Chronos max historical context
        context = torch.tensor(data)

        t1_start = time.perf_counter_ns()
        forecast = self.pipeline.predict(
            context,
            1, # prediction_length
            num_samples=20, # 20
            temperature=1.0, # 1.0
            top_k=64, # 50
            top_p=1.0, # 1.0
        ) # forecast shape: [num_series, num_samples, prediction_length]
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        predictions = median
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds

        print(f"Chronos prediction time seconds: {total_time}    Historical context size: {data.shape}    Predictions: {predictions}")
        return np.asarray(predictions, dtype=np.float32)

    # def score(self):
    #     pass


if __name__ == '__main__':
    test = ChronosAdapter()
    # test.predict()
