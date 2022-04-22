#!/usr/bin/env python
import sys
import numpy as np

emb_dim = int(sys.argv[1])

R = np.eye(emb_dim)

n, d = emb_dim, emb_dim
print(u"%d %d" % (n, d))
for i in range(n):
    print(" ".join(map(lambda a: "%.4f" % a, R[i, :])))
