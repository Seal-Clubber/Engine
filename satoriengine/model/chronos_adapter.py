import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from chronos import ChronosPipeline


class ChronosAdapter:

    def __init__(self, useGPU):
        hfhome = os.environ.get(
            'HF_HOME', default='/Satori/Neuron/models/huggingface')
        os.makedirs(hfhome, exist_ok=True)
        device_map = 'cuda' if useGPU else 'cpu'
        self.device = torch.device(device_map)
        self.pipeline = ChronosPipeline.from_pretrained(
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

        self.ctx_len = 512  # historical context
        self.pipeline.to(self.device)  # Move model to the appropriate device

    def fit(self, trainX, trainY, eval_set=None, verbose=False, epochs=10, batch_size=32, learning_rate=1e-4):
        # Convert training data to PyTorch tensors
        trainX_tensor = torch.tensor(
            trainX, dtype=torch.float32).to(self.device)
        trainY_tensor = torch.tensor(
            trainY, dtype=torch.float32).to(self.device)

        # Create a DataLoader for batching
        train_data = TensorDataset(trainX_tensor, trainY_tensor)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=4)

        # Define optimizer and loss function
        optimizer = optim.Adam(self.pipeline.parameters(),
                               lr=learning_rate, weight_decay=1e-5)
        loss_fn = nn.MSELoss()  # Assuming regression; you can change this if needed

        # Training loop
        for epoch in range(epochs):
            self.pipeline.train()  # Set model to training mode
            epoch_loss = 0
            for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                # Use the last `ctx_len` as context
                context = X_batch[:, -self.ctx_len:]
                context = torch.unsqueeze(context, -1)  # Add channel dimension
                forecast = self.pipeline(
                    context, prediction_length=1, num_samples=4)

                # Compute loss
                predictions = forecast.median(dim=1).values
                loss = loss_fn(predictions, Y_batch)
                epoch_loss += loss.item()

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / len(train_loader)
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            # Optionally: Evaluate on eval_set
            if eval_set is not None:
                self.evaluate(eval_set)

    def evaluate(self, eval_set):
        self.pipeline.eval()  # Set model to evaluation mode
        evalX, evalY = eval_set
        evalX_tensor = torch.tensor(evalX, dtype=torch.float32).to(self.device)
        evalY_tensor = torch.tensor(evalY, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Forward pass for evaluation
            context = evalX_tensor[:, -self.ctx_len:]
            context = torch.unsqueeze(context, -1)
            forecast = self.pipeline(
                context, prediction_length=1, num_samples=4)

            # Calculate loss or any other evaluation metric
            predictions = forecast.median(dim=1).values
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, evalY_tensor)
            print(f"Validation Loss: {loss.item():.4f}")

    def predict(self, current):
        data = current.values.astype(np.float32)
        data = np.squeeze(data, axis=0)
        data = data[-self.ctx_len:]
        context = np.pad(
            data, (self.ctx_len - data.shape[0], 0), mode='constant', constant_values=0)
        context = torch.tensor(context).to(self.device)

        t1_start = time.perf_counter_ns()
        forecast = self.pipeline.predict(
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

        print(
            f"Chronos prediction time seconds: {total_time}    Historical context size: {data.shape}    Predictions: {predictions}")
        return np.asarray(predictions, dtype=np.float32)


if __name__ == '__main__':
    test = ChronosAdapter(useGPU=False)
