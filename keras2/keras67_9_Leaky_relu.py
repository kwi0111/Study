# 난 정말 시그모이드
# SiLu (Sigmoid-weighted Linear Unit) = Swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def Leaky_ReLu(x):
    return np.maximum(0.01 * x, x)

y = Leaky_ReLu(x)


plt.plot(x, y)
plt.grid()
plt.show()

