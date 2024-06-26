import numpy as np
import torch
from chronos import ChronosPipeline

class ChronosAdapter():

    def __init__(self):
        self.pipeline = ChronosPipeline.from_pretrained(
            # "amazon/chronos-t5-tiny", # 8M
            # "amazon/chronos-t5-mini", # 20M
            "amazon/chronos-t5-small", # 46M
            # "amazon/chronos-t5-base", # 200M
            # "amazon/chronos-t5-large", # 710M
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            force_download=False,
        )
        self.trainX = None
        self.trainY = None


    def fit(self, trainX, trainY, eval_set, verbose):
        self.trainX = trainX
        self.trainY = trainY

    def predict(self, current):
        data = self.trainX.values
        data = np.squeeze(data)
        context = torch.tensor(data)
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

        return np.asarray(predictions, dtype=np.float32)

    # def score(self):
    #     pass


if __name__ == '__main__':
    test = ChronosAdapter()
    # test.predict()
