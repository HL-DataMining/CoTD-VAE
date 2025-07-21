# main.py

import torch
import torch.optim as optim
from model import CoTDVAE

def main():
    """
    An example script to demonstrate how to use the CoTD-VAE model.
    """
    # 1. Define hyperparameters and data dimensions
    batch_size = 32
    seq_len = 64
    input_dim = 9  # Number of features in the time series

    # 2. Create some dummy data
    # The shape should be [batch_size, input_dim, seq_len]
    print(f"Creating dummy data with shape: [{batch_size}, {input_dim}, {seq_len}]")
    dummy_data = torch.randn(batch_size, input_dim, seq_len)
    
    # 3. Instantiate the model
    print("Initializing CoTD-VAE model...")
    model = CoTDVAE(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=64,
        latent_static_dim=16,
        latent_trend_dim=16,
        latent_event_dim=16
    )

    # 4. Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Simple training loop
    print("\nStarting a simple training loop for 5 epochs...")
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        loss_dict, _, _, _ = model(dummy_data)
        total_loss = loss_dict['total_loss']
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Total Loss: {total_loss.item():.4f}")
        # Print individual loss components for inspection
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                print(f"  {loss_name}: {loss_value.item():.4f}")

    # 6. Demonstrate feature extraction
    print("\nDemonstrating feature extraction on the dummy data...")
    model.eval()
    extracted_features = model.extract_features(dummy_data)
    print(f"Shape of extracted features: {extracted_features.shape}")
    print("The extracted features can be used for downstream tasks like classification or clustering.")


if __name__ == "__main__":
    main()