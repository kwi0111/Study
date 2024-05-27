# 난 정말 시그모이드
# SiLu (Sigmoid-weighted Linear Unit) = Swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5)

def silu(x):
    return x * (1 / (1+np.exp(-x)))     # x * sigmoid(x)

y = silu(x)


plt.plot(x, y)
plt.grid()
plt.show()

