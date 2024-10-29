import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


class TTMAdapter():

    def __init__(self, useGPU):
        hfhome = os.environ.get(
            'HF_HOME', default='/Satori/Neuron/models/huggingface')
        os.makedirs(hfhome, exist_ok=True)
        device_map = 'cuda' if useGPU else 'cpu'
        self.device = torch.device(device_map)
        self.pipeline = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM",
            revision="main",  # 512 ctx, 96 pred
            # revision="1024_96_v1",  # 1024 ctx, 96 pred
            device_map=device_map,
            # force_download=True,
        )
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
        optimizer = optim.Adam(self.pipeline.parameters(), lr=learning_rate, weight_decay=1e-5)
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
                predictions = self.pipeline(context).prediction_outputs

                # Compute loss
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
            predictions = self.pipeline(context).prediction_outputs

            # Calculate loss or any other evaluation metric
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, evalY_tensor)
            print(f"Validation Loss: {loss.item():.4f}")

    def predict(self, current):
        data = current.values.astype(np.float32)
        data = np.squeeze(data, axis=0)
        data = data[-self.ctx_len:]
        context = np.pad(
            data, (self.ctx_len - data.shape[0], 0), mode='constant', constant_values=0)
        context = np.reshape(context, (1, -1, 1))
        context = torch.tensor(context).to(self.device)

        t1_start = time.perf_counter_ns()
        forecast = self.pipeline(context)
        predictions = forecast.prediction_outputs.detach().cpu().numpy()
        predictions = np.squeeze(predictions, axis=(0, -1))
        predictions = predictions[0:1]
        total_time = (time.perf_counter_ns() - t1_start) / 1e9  # seconds

        print(
            f"TTM prediction time seconds: {total_time}    Historical context size: {data.shape}    Predictions: {predictions}")
        return np.asarray(predictions, dtype=np.float32)


if __name__ == '__main__':
    test = TTMAdapter(useGPU=False)


comment = '''
To add training functionality to your TTMAdapter, you'll need to modify a few things:

Define the training loop.
Set the loss function and optimizer.
Process the data in batches for training.
Track the validation (evaluation) during training.
Here's how you can do that:

Step 1: Modify the fit method
You'll need to implement the fit method to allow the model to train on the provided trainX and trainY data. We'll use PyTorch's training loop structure to train the model.

Here's an updated version of your class with a training loop added:

Key Changes and Additions:
Data Preprocessing:

trainX and trainY are converted to PyTorch tensors and used to create a DataLoader for batching during training.
Training Loop:

The fit method implements a standard training loop:
A forward pass with the model's pipeline.
Loss calculation using mean squared error (MSE), though you can change this based on your specific use case (e.g., regression or classification).
Backpropagation with loss.backward() and optimizer step with optimizer.step() to update model weights.
Model Evaluation:

An optional evaluation step (eval_set) during training, using the same forward pass but with torch.no_grad() to disable gradient computation for evaluation. You can monitor loss on this evaluation set during training.
Batch Processing:

Batches of data are passed into the model during training, using the context length (self.ctx_len) to determine how much historical data to use for each prediction.
Parameters You Can Customize:
Epochs: The number of full passes through the training dataset.
Batch Size: Number of samples processed at a time during training.
Learning Rate: How quickly the model updates its weights.
With these changes, your TTMAdapter will now be able to train the TinyTimeMixer model on your specific data, instead of just using a pre-trained model.

'''
