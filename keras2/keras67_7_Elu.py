# 난 정말 시그모이드
# SiLu (Sigmoid-weighted Linear Unit) = Swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

alpha = 1
Elu = lambda x: np.where(x > 0, x * np.tanh(np.log(1 + np.exp(0))), alpha * (np.exp(x) - 1) * np.tanh(np.log(1 + np.exp(0))))

y = Elu(x)


plt.plot(x, y)
plt.grid()
plt.show()

