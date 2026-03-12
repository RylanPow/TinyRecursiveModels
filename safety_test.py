import os
import torch

checkpoint_dir = "/workspace/TinyRecursiveModels/checkpoints/SpaceCheck"
os.makedirs(checkpoint_dir, exist_ok=True)

print("Testing write permissions and disk space...")

# Generate a fake 300MB tensor
fake_weights = torch.randn(75_000_000) 

try:
    test_file = os.path.join(checkpoint_dir, "massive_dummy_checkpoint.pt")
    torch.save(fake_weights, test_file)
    file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
    print(f"SUCCESS: Wrote a {file_size_mb:.2f} MB test file to the volume disk.")
    
    os.remove(test_file)
    print("Test file removed. Your volume disk is safe for the experiments.")
except OSError as e:
    print(f"FAILURE: Disk issue detected! Error: {e}")