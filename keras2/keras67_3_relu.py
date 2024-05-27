# 난 정말 시그모이드

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

relu = lambda  x : np.maximum(0, x)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

