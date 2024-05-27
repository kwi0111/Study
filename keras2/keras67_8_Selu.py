# 난 정말 시그모이드
# SiLu (Sigmoid-weighted Linear Unit) = Swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
alpha = 1.6732632423543772848170429916717
lambda_param = 1.0507009873554804934193349852946

Selu = lambda x: np.where(x > 0, lambda_param * x, lambda_param * alpha * (np.exp(x) - 1))

y = Selu(x)

plt.plot(x, y)
plt.grid()
plt.show()

