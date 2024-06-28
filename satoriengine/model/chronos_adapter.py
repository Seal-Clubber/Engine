import numpy as np
import torch
from chronos import ChronosPipeline
import time

class ChronosAdapter():

    def __init__(self):
        self.pipeline = ChronosPipeline.from_pretrained(
            # "amazon/chronos-t5-tiny", # 8M
            # "amazon/chronos-t5-mini", # 20M
            "amazon/chronos-t5-small", # 46M
            # "amazon/chronos-t5-base", # 200M
            # "amazon/chronos-t5-large", # 710M
            device_map="cpu", # "cpu" for any CPU, "cuda" for Nvidia GPU, "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
            force_download=False,
        )
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None


    def fit(self, trainX, trainY, eval_set, verbose):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = eval_set[1][0]
        self.testY = eval_set[1][1]

    def predict(self, current):
        data = self.testY.values
        data = np.squeeze(data)
        data = data[-512:] # Chronos max historical context
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
        print(f"Chronos prediction time seconds: {total_time}    Historical context size: {data.shape}")

        return np.asarray(predictions, dtype=np.float32)

    # def score(self):
    #     pass


if __name__ == '__main__':
    test = ChronosAdapter()
    # test.predict()
