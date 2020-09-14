# Fast Pytorch Kmeans
this is a pytorch implementation of K-means clustering algorithm

# Quick Start
```python
from kmeans import KMeans
import torch

kmeans = KMeans(n_clusters=8, mode='euclidean', verbose=1)
x = torch.randn(100000, 64, device='cuda')
labels = kmeans.fit_predict(x)
```

# Speed Comparison
![](./img/1.png)
![](./img/2.png)
![](./img/3.png)
