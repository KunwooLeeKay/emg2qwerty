import torch

# Load the checkpoint
checkpoint_path = "logs/2025-03-15/12-16-38/job0_trainer.devices=8,user=generic/checkpoints/epoch=144-step=129920.ckpt"
checkpoint = torch.load(checkpoint_path)

# Print available keys in the checkpoint
print("Keys in checkpoint:", checkpoint.keys())

# Check the logged metrics
if "callbacks" in checkpoint:
    print("Saved metrics:", checkpoint["callbacks"])
elif "monitor" in checkpoint:
    print("Monitoring metric:", checkpoint["monitor"])
else:
    print("No saved metrics found in checkpoint.")
