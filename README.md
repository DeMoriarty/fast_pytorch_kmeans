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
<p>Tested on google colab with Intel(R) Xeon(R) CPU @ 2.00GHz and Nvidia Tesla T4 GPU</p>

<h4> sklearn: sklearn.cluster.KMeans</h4>
<ul>
 <li>n_init = 1</li>
 <li>max_iter = 100</li>
 <li>tol = -1 (to force 100 iterations)</li>
</ul>

<h4> faiss: faiss.Clustering </h4>
<ul>
 <li>nredo = 1</li>
 <li>niter = 100</li>
 <li>max_point_per_centroid = 10**9 (to prevent subsample from dataset) </li>
</ul>
<p>note: time cost for transfering data from cpu to gpu is also included </p>

<h4> fast-pytorch: fast_pytorch_kmeans.KMeans </h4>
<ul>
 <li>max_iter = 100 </li>
 <li>tol = -1 (to force 100 iterations)</li>
 <li>minibatch = None </li>
</ul>

### 1. n_samples=100,000, n_features=256, time spent for 100 iterations
<p float="left">
  <img src="/img/fig1.png" width="49%"/>
  <img src="/img/semilog1.png" width="50%" /> 
</p>

### 2. n_samples=100,000, n_clusters=256, time spent for 100 iterations
<p float="left">
  <img src="/img/fig2.png" width="49%"/>
  <img src="/img/semilog2.png" width="50%" /> 
</p>

### 3. n_features=256, n_clusters=256, time spent for 100 iterations
<p float="left">
  <img src="/img/fig3.png" width="49%"/>
  <img src="/img/semilog3.png" width="50%" /> 
</p>

### 4. n_features=32, n_clusters=1024, time spent for 100 iterations
<p float="left">
  <img src="/img/fig4.png" width="49%"/>
  <img src="/img/semilog4.png" width="50%" /> 
</p>

### 5. n_features=1024, n_clusters=32, time spent for 100 iterations
<p float="left">
  <img src="/img/fig5.png" width="49%"/>
  <img src="/img/semilog5.png" width="50%" /> 
</p>
