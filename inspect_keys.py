import sys
import mlx.core as mx
from models.trm_arc import ARCModel, ARCModelConfig

path = sys.argv[1]
config = ARCModelConfig(vocab_size=12, max_seq_len=900, depth=2, dim=128, heads=8)
model = ARCModel(config)
model.load_weights(path)

for name, param in model.parameters().items():
    print('param', name, param.shape)
for name, param in model.trainable_parameters().items():
    print('trainable', name, param.shape)
