![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
# CoTD-VAE: Constrained Temporal Disentanglement VAE

This repository contains a PyTorch implementation of a **Constrained Temporal Disentanglement Variational Autoencoder (CoTD-VAE)**. The model is designed to decompose multivariate time series data into three interpretable latent components: a **static** component, a **trend** component, and an **event** component.

## Core Idea

The key innovation of this model lies in its specialized loss functions that guide the disentanglement process:

-   **Static Component (`z_static`)**: Captures time-invariant, global properties of the entire series. It is encoded by an LSTM that sees the whole sequence.
-   **Trend Component (`z_trend`)**: Represents the smooth, slowly-varying dynamics of the series. A smoothness loss (`L_smooth`) is applied to penalize sharp changes in this latent space, enforcing a smooth representation.
-   **Event Component (`z_event`)**: Models sparse, sudden bursts or anomalies. A sparsity loss (`L_sparse`) encourages this latent representation to be mostly zero, with sharp, high-magnitude peaks at specific time points.

By training the VAE to reconstruct the original series while simultaneously optimizing these specialized losses, the model learns to automatically disentangle the underlying factors of variation.

## Project Structure

cotd-vae-project/

├── .gitignore

├── LICENSE

├── README.md

├── requirements.txt

├── model.py

└── main.py

## Installation

Install the required dependencies. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
*Note: `torch` is the only dependency. For GPU support, please follow the official PyTorch installation instructions for your specific CUDA version.*

## Usage

The `main.py` script provides a simple, self-contained example of how to use the `CoTDVAE` model. It creates dummy data, initializes the model, runs a few training steps, and demonstrates how to extract features.

To run the example:
```bash
python main.py
```

You should see output similar to this:

Creating dummy data with shape: [32, 9, 64]
Initializing CoTD-VAE model...

Starting a simple training loop for 5 epochs...

--- Epoch 1/5 ---

Total Loss: 296.7708

  recon_loss: 1.0632
  
  kl_static: 0.0431
  
  kl_trend: 138.9658
  
  kl_event: 150.7921
  
  L_smooth: 5.9330
  
  L_sparse: -0.0264
... (and so on for other epochs)

Demonstrating feature extraction on the dummy data...
Shape of extracted features: torch.Size([32, 48])
The extracted features can be used for downstream tasks like classification or clustering.
