from lightning import TDSConvCTCModule

def test_cpu_model():
    import torch

    # Dummy input
    dummy_inputs = torch.randn(16, 2, 2, 16, 10)

    model = TDSConvCTCModule(
        in_features=16 * 10,  # or appropriate
        mlp_features=[64, 128],
        block_channels=[32, 32],
        kernel_width=8,
        optimizer=None,  # or dummy
        lr_scheduler=None,  # or dummy
        decoder=None,  # or dummy
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
    ).cpu()  

    outputs = model(dummy_inputs)
    print("Output shape:", outputs.shape)

# Then run
test_cpu_model()
