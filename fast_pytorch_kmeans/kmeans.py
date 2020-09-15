import math
import torch
from time import time
import numpy as np

class KMeans:
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
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

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
    batch_size = a.shape[0]
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim

    if device == 'cpu':
      sim = sim_func(a, b)
      max_sim_v, max_sim_i = sim.max(dim=-1)
      return max_sim_v, max_sim_i
    else:
      if a.dtype == torch.float:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
      elif a.dtype == torch.half:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
      ratio = math.ceil(expected / self.remaining_memory())
      subbatch_size = math.ceil(batch_size / ratio)
      msv, msi = [], []
      for i in range(ratio):
        if i*subbatch_size >= batch_size:
          continue
        sub_x = a[i*subbatch_size: (i+1)*subbatch_size]
        sub_sim = sim_func(sub_x, b)
        sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        del sub_sim
        msv.append(sub_max_sim_v)
        msi.append(sub_max_sim_i)
      if ratio == 1:
        max_sim_v, max_sim_i = msv[0], msi[0]
      else:
        max_sim_v = torch.cat(msv, dim=0)
        max_sim_i = torch.cat(msi, dim=0)
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
    batch_size, emb_dim = X.shape
    device = X.device.type
    start_time = time()
    if centroids is None:
      self.centroids = X[np.random.choice(batch_size, size=[self.n_clusters], replace=False)]
    else:
      self.centroids = centroids
    num_points_in_clusters = torch.ones(self.n_clusters, device=device)
    closest = None
    for i in range(self.max_iter):
      iter_time = time()
      if self.minibatch is not None:
        x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
      else:
        x = X
      closest = self.max_sim(a=x, b=self.centroids)[1]
      matched_clusters, counts = closest.unique(return_counts=True)

      c_grad = torch.zeros_like(self.centroids)
      if self._loop:
        for j, count in zip(matched_clusters, counts):
          c_grad[j] = x[closest==j].sum(dim=0) / count
      else:
        if self.minibatch is None:
          expanded_closest = closest[None].expand(self.n_clusters, -1)
          mask = (expanded_closest==torch.arange(self.n_clusters, device=device)[:, None]).float()
          c_grad = mask @ x / mask.sum(-1)[..., :, None]
        else:
          expanded_closest = closest[None].expand(len(matched_clusters), -1)
          mask = (expanded_closest==matched_clusters[:, None]).float()
          c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]


        # if x.dtype == torch.float:
        #   expected = closest.numel() * len(matched_clusters) * 5 # bool+float
        # elif x.dtype == torch.half:
        #   expected = closest.numel() * len(matched_clusters) * 3 # bool+half
        # if device == 'cpu':
        #   ratio = 1
        # else:
        #   ratio = math.ceil(expected / self.remaining_memory() )
        # # ratio = 1
        # subbatch_size = math.ceil(len(matched_clusters)/ratio)
        # for j in range(ratio):
        #   if j*subbatch_size >= batch_size:
        #     continue
        #   sub_matched_clusters = matched_clusters[j*subbatch_size: (j+1)*subbatch_size]
        #   sub_expanded_closest = closest[None].expand(len(sub_matched_clusters), -1)
        #   sub_mask = (sub_expanded_closest==sub_matched_clusters[:, None]).to(x.dtype)
        #   sub_prod = sub_mask @ x / sub_mask.sum(1)[:, None]
        #   c_grad[sub_matched_clusters] = sub_prod
      error = (c_grad - self.centroids).pow(2).sum()
      if self.minibatch is not None:
        lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
        # lr = 1/num_points_in_clusters[:,None]**0.1 
      else:
        lr = 1
      num_points_in_clusters[matched_clusters] += counts
      self.centroids = self.centroids * (1-lr) + c_grad * lr
      if self.verbose >= 2:
        print('iter:', i, 'error:', error.item(), 'time spent:', round(time()-iter_time, 4))
      if error <= self.tol:
        break

    # SCATTER
    if self._show:
      if self.mode is "cosine":
        sim = self.cos_sim(x, self.centroids)
      elif self.mode is "euclidean":
        sim = self.euc_sim(x, self.centroids)
      closest = sim.argmax(dim=-1)
      plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=closest.cpu(), marker='.', cmap='hsv')
      # plt.scatter(c[:,0].cpu(), c[:,1].cpu(), marker='o', cmap='red')
      plt.show()
    # END SCATTER
    if self.verbose >= 1:
      print(f'used {i+1} iterations ({round(time()-start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters')
    return closest

  def predict(self, X):
    """
      Predict the closest cluster each sample in X belongs to

      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]

      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    return self.max_sim(a=X, b=self.centroids)[1]

  def fit(self, X, centroids=None):
    """
      Perform kmeans clustering

      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
    """
    self.fit_predict(X, centroids)