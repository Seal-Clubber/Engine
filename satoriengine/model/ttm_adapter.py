import time
import numpy as np
import torch
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

class TTMAdapter():

    def __init__(self, useGPU):
        device_map = 'cuda' if useGPU else 'cpu'
        self.pipeline = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM",
            revision="main", # 512-96
            # revision="1024_96_v1",
            device_map=device_map,
            # force_download=True,
        )
        self.ctx_len = 512 # historical context

    def fit(self, trainX, trainY, eval_set, verbose):
        pass

    def predict(self, current):
        data = current.values.astype(np.float32)
        data = np.squeeze(data)
        data = data[-self.ctx_len:]
        data = np.pad(data, (self.ctx_len - data.shape[0], 0), mode='constant', constant_values=0)
        data = np.reshape(data, (1,-1,1))
        context = torch.tensor(data)

        t1_start = time.perf_counter_ns()
        forecast = self.pipeline(context)
        predictions = forecast.prediction_outputs.detach().numpy()
        predictions = np.squeeze(predictions)
        predictions = predictions[0]
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds

        print(f"TTM prediction time seconds: {total_time}    Historical context size: {data.shape}    Predictions: {predictions}")
        return np.asarray(predictions, dtype=np.float32)

    # def score(self):
    #     pass


if __name__ == '__main__':
    test = TTMAdapter()
    # test.predict()
