import torch
import torch.nn as nn

model = nn.LSTM(128, 256, 3).cuda()
x = torch.randn(64, 20, 128).cuda()

for _ in range(20):
    y, _ = model(x)

print("OK:", y.shape)