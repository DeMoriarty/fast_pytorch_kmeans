from distutils.core import setup
setup(
  name = 'fast_pytorch_kmeans', 
  packages = ['fast_pytorch_kmeans'],
  version = '0.2.0.1', 
  license='MIT',
  description = 'a fast kmeans clustering algorithm implemented in pytorch',
  author = 'demoriarty', 
  author_email = 'sahbanjan@gmail.com', 
  url = 'https://github.com/DeMoriarty/fast_pytorch_kmeans',
  download_url = 'https://github.com/DeMoriarty/fast_pytorch_kmeans/archive/v_020.tar.gz',
  keywords = ['KMeans', 'K-means', 'pytorch','machine learning'],
  install_requires=[ 
          'numpy',
          'torch',
          'nvidia-ml-py'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3', 
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)