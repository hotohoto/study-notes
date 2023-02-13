# GAN (Generative Adversarial Network)

- GAN
  - Image -> D -> D(x)
  - z -> G -> Image -> D
  - D: binary classification: Fake(0) or Real(1)
  - G: image generation
    - wants D classifies generated images as 1
    - z is random sampled vector from a distribution like gaussian

https://github.com/yunjey/pytorch-tutorial
https://youtu.be/odpjk7_tGY0

pseudo code

```py
import torch
import torch.nn as nn

d = nn.Sequential(
    nn.linear(784, 128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid()
)

g = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Tanh()
)

criterion = nn.BCELoss()

d_optimizer = torch.optim.adam(d.parameters(), lr=0.0002)
g_optimizer = torch.optim.adam(g.parameters(), lr=0.0002)

while True:
    loss = criterion(d(x), 1) + criterion(d(g(x)), 0)
    loss.backward()
    d_optimizer.step()

    loss = criterion(d(g(x)), 1)
    loss.backward()
    g_optimizer.step()
```

## Tips

well-known adam parameter

- lr=0.0002
- beta1=0.5
- beta2=0.999
