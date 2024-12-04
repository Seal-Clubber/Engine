from typing import Union
import os
import time
import numpy as np
import torch
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult
from satorilib.logging import debug
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from chronos import ChronosPipeline

class ChronosVedaPipeline(PipelineInterface):

    def __init__(self, useGPU):
        hfhome = os.environ.get(
            'HF_HOME',
            default='/Satori/Neuron/models/huggingface')
        os.makedirs(hfhome, exist_ok=True)
        deviceMap = 'cuda' if useGPU else 'cpu'
        self.device = torch.device(deviceMap)
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large" if useGPU else "amazon/chronos-t5-small",
            # "amazon/chronos-t5-tiny", # 8M
            # "amazon/chronos-t5-mini", # 20M
            # "amazon/chronos-t5-small", # 46M
            # "amazon/chronos-t5-base", # 200M
            # "amazon/chronos-t5-large", # 710M
            # 'cpu' for any CPU, 'cuda' for Nvidia GPU, 'mps' for Apple Silicon
            device_map=deviceMap,
            torch_dtype=torch.bfloat16,
            # force_download=True,
        )
        self.ctx_len = 512  # historical context
        self.ctx_len = 512  # historical context
        self.model.model.to(self.device)  # Move model to the appropriate device

    def fit(
        self,
        trainX,
        trainY,
        eval_set=None,
        verbose=False,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
    ):
        # Convert training data to PyTorch tensors
        trainX_tensor = torch.tensor(
            trainX,
            dtype=torch.float32
        ).to(self.device)
        trainY_tensor = torch.tensor(
            trainY,
            dtype=torch.float32
        ).to(self.device)

        # Create a DataLoader for batching
        train_data = TensorDataset(trainX_tensor, trainY_tensor)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

        # Define optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5)
        loss_fn = nn.MSELoss()  # Assuming regression; you can change this if needed

        # Training loop
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            epoch_loss = 0
            for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                # Use the last `ctx_len` as context
                context = X_batch[:, -self.ctx_len:]
                context = torch.unsqueeze(context, -1)  # Add channel dimension
                forecast = self.model(
                    context,
                    prediction_length=1,
                    num_samples=4)

                # Compute loss
                predictions = forecast.median(dim=1).values
                loss = loss_fn(predictions, Y_batch)
                epoch_loss += loss.item()

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / len(train_loader)
            if verbose:
                debug(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            # Optionally: Evaluate on eval_set
            if eval_set is not None:
                self.evaluate(eval_set)

        return TrainingResult(1, self, False)

    def evaluate(self, eval_set):
        self.model.eval()  # Set model to evaluation mode
        evalX, evalY = eval_set
        evalX_tensor = torch.tensor(evalX, dtype=torch.float32).to(self.device)
        evalY_tensor = torch.tensor(evalY, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Forward pass for evaluation
            context = evalX_tensor[:, -self.ctx_len:]
            context = torch.unsqueeze(context, -1)
            forecast = self.model(
                context, prediction_length=1, num_samples=4)

            # Calculate loss or any other evaluation metric
            predictions = forecast.median(dim=1).values
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, evalY_tensor)
            debug(f"Validation Loss: {loss.item():.4f}")

    def predict(self, current):
        data = current.values.astype(np.float32)
        data = np.squeeze(data, axis=0)
        data = data[-self.ctx_len:]
        context = np.pad(
            data, (self.ctx_len - data.shape[0], 0), mode='constant', constant_values=0)
        context = torch.tensor(context).to(self.device)
        t1_start = time.perf_counter_ns()
        forecast = self.model.predict(
            context,
            1,  # prediction_length
            num_samples=4,  # 20
            temperature=1.0,  # 1.0
            top_k=64,  # 50
            top_p=1.0,  # 1.0
        )  # forecast shape: [num_series, num_samples, prediction_length]
        median = forecast.median(dim=1).values
        predictions = median[0]
        total_time = (time.perf_counter_ns() - t1_start) / 1e9  # seconds
        debug(
            f"Chronos prediction time seconds: {total_time}"
            f"\nHistorical context size: {data.shape}"
            f"\nPredictions: {predictions}")
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
