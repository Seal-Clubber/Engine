import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


class TTMAdapter:

    def __init__(self, useGPU):
        hfhome = os.environ.get('HF_HOME', default='/Satori/Neuron/models/huggingface')
        os.makedirs(hfhome, exist_ok=True)
        device_map = 'cuda' if useGPU and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_map)
        self.ctx_len = 512
        self.pipeline = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM",
            revision="main",  # 512 ctx, 96 pred
            device_map=device_map,
        )
        print(f"✅ Using device: {self.device}")

    def _pad_or_truncate(self, data, target_len):
        """Ensure each sample is exactly `target_len` long (pad or trim)."""
        padded = []
        for row in data:
            row = np.asarray(row, dtype=np.float32).flatten()
            if len(row) < target_len:
                row = np.pad(row, (target_len - len(row), 0), mode='constant')
            else:
                row = row[-target_len:]
            padded.append(row)
        return np.array(padded, dtype=np.float32)

    def fit(self, trainX, trainY, eval_set=None, verbose=False, epochs=10, batch_size=32, learning_rate=1e-4):
        if not isinstance(trainX, (list, np.ndarray)):
            raise TypeError(f"❌ trainX must be a list or array. Got: {type(trainX)}")

        trainX = self._pad_or_truncate(trainX, self.ctx_len)
        trainY = np.array(trainY, dtype=np.float32)
        if trainY.ndim == 2:
            trainY = np.expand_dims(trainY, axis=-1)  # Shape (N, 96, 1)

        trainX_tensor = torch.tensor(trainX, dtype=torch.float32).to(self.device)
        trainY_tensor = torch.tensor(trainY, dtype=torch.float32).to(self.device)

        train_data = TensorDataset(trainX_tensor, trainY_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

        self.pipeline.to(self.device)
        optimizer = optim.Adam(self.pipeline.parameters(), lr=learning_rate, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.pipeline.train()
            epoch_loss = 0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                context = X_batch[:, -self.ctx_len:]
                context = torch.unsqueeze(context, -1)
                predictions = self.pipeline(context).prediction_outputs
                loss = loss_fn(predictions, Y_batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / len(train_loader)
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            if eval_set is not None:
                self.evaluate(eval_set)

    def evaluate(self, eval_set):
        self.pipeline.eval()
        evalX, evalY = eval_set
        evalX = self._pad_or_truncate(evalX, self.ctx_len)
        evalY = np.array(evalY, dtype=np.float32)
        if evalY.ndim == 2:
            evalY = np.expand_dims(evalY, axis=-1)

        evalX_tensor = torch.tensor(evalX, dtype=torch.float32).to(self.device)
        evalY_tensor = torch.tensor(evalY, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            context = evalX_tensor[:, -self.ctx_len:]
            context = torch.unsqueeze(context, -1)
            predictions = self.pipeline(context).prediction_outputs

            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, evalY_tensor)
            print(f"Validation Loss: {loss.item():.4f}")

    def predict(self, current):
        data = current.values.astype(np.float32).flatten()
        data = data[-self.ctx_len:]
        context = np.pad(data, (self.ctx_len - len(data), 0), mode='constant')
        context = np.reshape(context, (1, -1, 1))
        context_tensor = torch.tensor(context, dtype=torch.float32).to(self.device)

        self.pipeline.to(self.device)
        t1_start = time.perf_counter_ns()
        forecast = self.pipeline(context_tensor)
        predictions = forecast.prediction_outputs.detach().cpu().numpy()
        predictions = np.squeeze(predictions, axis=(0, -1))
        predictions = predictions[0:1]
        total_time = (time.perf_counter_ns() - t1_start) / 1e9

        print(f"⏱️ Inference time: {total_time:.4f}s | Predictions: {predictions}")
        return np.asarray(predictions, dtype=np.float32)


if __name__ == '__main__':
    test = TTMAdapter(useGPU=True)
