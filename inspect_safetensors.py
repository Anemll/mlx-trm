import safetensors.torch
import sys

path = sys.argv[1]
weights = safetensors.torch.load_file(path)
for name in weights.keys():
    print(name)
