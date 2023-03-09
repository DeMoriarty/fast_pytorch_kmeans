import math
import torch
from time import time
import numpy as np

class MultiKMeans:
  '''
  Kmeans clustering algorithm implemented with PyTorch
  Parameters:
    n_clusters: int, 
      Number of clusters
    max_iter: int, default: 100
      Maximum number of iterations
    tol: float, default: 0.0001
      Tolerance
    
    verbose: int, default: 0
      Verbosity
    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      Type of distance measure
    minibatch: {None, int}, default: None
      Batch size of MinibatchKmeans algorithm
      if None perform full KMeans algorithm
      
  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      cluster centroids
  '''
  def __init__(self, n_clusters, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", minibatch=None):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.mode = mode
    self.minibatch = minibatch
    self._loop = False
    self._show = False

    try:
      import PYNVML
      self._pynvml_exist = True
    except ModuleNotFoundError:
      self._pynvml_exist = False
    
    self.centroids = None

  @staticmethod
  def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    a_norm = a.norm(dim=-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True)
    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)
    return a @ b.transpose(-2, -1)

  @staticmethod
  def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=-1)[..., :, None] - (b**2).sum(dim=-1)[..., None, :]

  def remaining_memory(self):
    """
      Get remaining memory in gpu
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if self._pynvml_exist:
      pynvml.nvmlInit()
      gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
      remaining = info.free
    else:
      remaining = torch.cuda.memory_allocated()
    return remaining

  def max_sim(self, a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in a with all of the vectors in b
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    device = a.device.type
    n_samples = a.shape[-2]
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim

    sim = sim_func(a, b)
    max_sim_v, max_sim_i = sim.max(dim=-1)
    return max_sim_v, max_sim_i


  def fit_predict(self, X, centroids=None):
    """
      Combination of fit() and predict() methods.
      This is faster than calling fit() and predict() seperately.
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
      centroids: {torch.Tensor, None}, default: None
        if given, centroids will be initialized with given tensor
        if None, centroids will be randomly chosen from X
      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    n_kmeans, n_samples, n_features = X.shape
    self.n_kmeans = n_kmeans

    device = X.device.type
    start_time = time()
    if self.centroids is None:
      self.centroids = X[:, np.random.choice(n_samples, size=[self.n_clusters], replace=False)]

    if centroids is not None:
      self.centroids = centroids
    num_points_in_clusters = torch.ones(self.n_kmeans, self.n_clusters, device=device, dtype=X.dtype)
    closest = None
    for i in range(self.max_iter):
      iter_time = time()
      if self.minibatch is not None:
        x = X[:, np.random.choice(n_samples, size=[self.minibatch], replace=False)]
      else:
        x = X
      closest = self.max_sim(a=x, b=self.centroids)[1]
      uniques = [closest[i].unique(return_counts=True) for i in range(self.n_kmeans)]
      c_grad = torch.zeros_like(self.centroids)

      expanded_closest = closest[:, None].expand(-1, self.n_clusters, -1)
      mask = (expanded_closest==torch.arange(self.n_clusters, device=device)[None, :, None]).to(X.dtype)
      c_grad = mask @ x / mask.sum(-1, keepdim=True)
      c_grad[c_grad!=c_grad] = 0 # remove NaNs

      error = (c_grad - self.centroids).pow(2).sum()
      if self.minibatch is not None:
        lr = 1/num_points_in_clusters[:,:,None] * 0.9 + 0.1
      else:
        lr = 1
      for j in range(self.n_kmeans):
        num_points_in_clusters[j, uniques[j][0]] += uniques[j][1]
      self.centroids = self.centroids * (1-lr) + c_grad * lr
      if self.verbose >= 2:
        print('iter:', i, 'error:', error.item(), 'time spent:', round(time()-iter_time, 4))
      if error <= self.tol * self.n_kmeans:
        break

    if self.verbose >= 1:
      print(f'used {i+1} iterations ({round(time()-start_time, 4)}s) to cluster {self.n_kmeans}x{n_samples} items into {self.n_clusters} clusters')
    return closest

  def predict(self, X):
    """
      Predict the closest cluster each sample in X belongs to
      Parameters:
      X: torch.Tensor, shape: [n_kmeans, n_samples, n_features]
      Return:
      labels: torch.Tensor, shape: [n_kmeans, n_samples]
    """
    return self.max_sim(a=X, b=self.centroids)[1]

  def fit(self, X, centroids=None):
    """
      Perform kmeans clustering
      Parameters:
      X: torch.Tensor, shape: [n_kmeans, n_samples, n_features]
    """
    self.fit_predict(X, centroids)
