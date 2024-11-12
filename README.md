# lrv-test
Implementation of the Loubaton-Rosuel-Vallet test

```
from lrv_test import LRV
import numpy as np 

y = np.arange(100*10).reshape(100, 10)
B = 21 
f = lambda x: (x - 1) ** 2
lrv_results = LRV(y, B, f, L=3)
lrv_results.t_stat_3, lrv_results.is_positive_3(level=0.05)
```