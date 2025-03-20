from lightning import TDSConvCTCModule

import torch

def test_gpu_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy input tensor
    dummy_inputs = torch.randn(16, 2, 2, 16, 10).to(device)  # Move inputs to GPU

    # Instantiate model and move to GPU
    model = TDSConvCTCModule(
        in_features=160,
        mlp_features=[64, 128],
        block_channels=[32, 32],
        kernel_width=8,
        optimizer=None,
        lr_scheduler=None,
        decoder=None,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
    ).to(device)  # Move model to GPU

    # Run forward pass
    outputs = model(dummy_inputs)
    print("GPU Output shape:", outputs.shape)

# Run test
test_gpu_model()

import torch
import torch.optim as optim

def test_gpu_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model & move to GPU
    model = TDSConvCTCModule(
        in_features=160,
        mlp_features=[64, 128],
        block_channels=[32, 32],
        kernel_width=8,
        optimizer=None,
        lr_scheduler=None,
        decoder=None,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create dummy input & target (for CTC loss)
    T, N, num_classes = 16, 2, 99
    inputs = torch.randn(T, N, 2, 16, 10).to(device)*0.01  # Move input to GPU
    # input_lengths = torch.full((N,), T, dtype=torch.long).to(device)  # Assume full sequence length
    


    # Forward pass
    outputs = torch.nan_to_num(model(inputs), nan=0.0, posinf=1.0, neginf=-1.0)

    print("GPU Training Output shape:", outputs.shape)
    # Get the actual output time dimension (T') after TDSConvEncoder & Transformer
    T_prime = outputs.shape[0]  # Should be 2 in your case
    target_lengths = torch.randint(1, T_prime + 1, (N,), dtype=torch.long).to(device)  # Ensure valid target lengths
    # Generate target sequences of varying lengths
    targets = [torch.randint(1, num_classes, (l,), dtype=torch.long).to(device) for l in target_lengths.tolist()]

    # Concatenate all targets into a 1D tensor
    targets = torch.cat(targets)

    # Set input_lengths to match the actual output time dimension
    input_lengths = torch.full((N,), T_prime, dtype=torch.long).to(device)

    # Compute loss
    ctc_loss = torch.nn.CTCLoss(blank=0)
    loss = ctc_loss(outputs.log_softmax(2), targets, input_lengths, target_lengths)
    print("CTC Loss:", loss.item())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
    optimizer.step()

    print("GPU Training Step Successful!")

# Run test
test_gpu_training()
