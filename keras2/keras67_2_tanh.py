# 난 정말 시그모이드

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)
print((x))
print(len(x))   # 100


plt.plot(x, y)
plt.grid()
plt.show()

