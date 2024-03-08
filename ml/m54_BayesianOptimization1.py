

param_bounds = {'x1' : (-1, 5),
                'x2' : (0, 4),
                }

def y_function(x1, x2):
    return -x1 **2 - (x2 - 2) **2 + 10

# pip install bayesian-optimization로 설치 해야한다. // 자동 튜닝중에 가장 약한놈.
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f= y_function,   # 최대값을 찾겠다.
    pbounds=param_bounds,
    random_state=87979,
)

optimizer.maximize(init_points=5,   # n개 찍을거야
                   n_iter=20,       # n만큼 훈련 == 25번 훈련(?)
                   )

print(optimizer.max)
