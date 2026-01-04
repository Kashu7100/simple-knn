simple-knn
---

>⚠️ LICENSE WARNING: This project is a derivative of simple-knn


Description: It compute the **average distance to the nearest neighbors** for a set of 3D points.

Install:
```bash
pip install git+https://github.com/camenduru/simple-knn
```

Usage:
```python
from simple_knn import distCUDA2

# shape: [N, 3]
demopc = torch.from_numpy(np.load("/path")).float().cuda().contiguous() 

# shape: [N]
mean_distances = distCUDA2(demopc)
```
