# Fast Pytorch Kmeans
this is a pytorch implementation of K-means clustering algorithm

## Installation
```
pip install fast-pytorch-kmeans
```

## Quick Start
```python
from fast_pytorch_kmeans import KMeans
import torch

kmeans = KMeans(n_clusters=8, mode='euclidean', verbose=1)
x = torch.randn(100000, 64, device='cuda')
labels = kmeans.fit_predict(x)
```

## Speed Comparison
<p>Tested on google colab with a Tesla K80 GPU</p>
<h5> sklearn: sklearn.cluster.KMeans (n_init=1)</h5>
<h5> faiss: faiss.Clustering (nredo=1)</h5>
<h5> fast-pytorch: fast_pytorch_kmeans.KMeans </h5>

n_samples=100,000 n_features=64 iterations=100
<p float="left">
  <img src="/img/fig1.png" width="49%"/>
  <img src="/img/semilog1.png" width="50%" /> 
</p>

n_samples=100,000 n_clusters=64 iterations=100
<p float="left">
  <img src="/img/fig2.png" width="49%"/>
  <img src="/img/semilog2.png" width="50%" /> 
</p>

n_features=256, n_clusters=256 iterations=100
<p float="left">
  <img src="/img/fig3.png" width="49%"/>
  <img src="/img/semilog3.png" width="50%" /> 
</p>

n_features=512, n_clusters=512 iterations=100
<p float="left">
  <img src="/img/fig4.png" width="49%"/>
  <img src="/img/semilog4.png" width="50%" /> 
</p>
