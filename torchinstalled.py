import torch

import torch

print("PyTorch version:", torch.__version__)
print("CUDA (GPU) Available:", torch.cuda.is_available())

# Create a simple tensor to verify PyTorch functionality
tensor = torch.tensor([1.0, 2.0, 3.0])
print("Test Tensor:", tensor)