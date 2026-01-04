simple-knn
---

> ⚠️ LICENSE WARNING: This project is a derivative of simple-knn and is licensed under the GRAPHDECO research group.


Description: It compute the **average distance to the nearest neighbors** for a set of 3D points.

Install:
```bash
pip install eden-simple-knn==0.1.0a0
```

Usage:
```python
from simple_knn import distCUDA2

# shape: [N, 3]
demopc

# shape: [N]
mean_distances = distCUDA2(demopc)
```
