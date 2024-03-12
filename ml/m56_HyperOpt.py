# import numpy as np
# import hyperopt
# print (hyperopt.__version__)  # 0.2.7
# from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
# import pandas as pd

# search_space = {'x1' : hp.quniform('x1', -10, 10, 1),        # hp.quniform: 균등 분포에서 값을 샘플링하는 함수
#                 'x2' : hp.quniform('x2', -15, 15, 1)}
# # hp.quniform(label, low, high, q) : 최소값 low에서 최대값 high까지 q의 간격을 가지고 설정
# # hp.uniform(label, low, high) : 최소값 low에서 최대값 high까지 정규분포 형태의 검색 공간 설정
# # hp.randint(label, upper) : 0부터 최대값 upper까지 random한 정수값으로 검색 공간 설정.
# # hp.loguniform(label, low, high) : exp(uniform(low, high))값을 반환하며,
# #                                   반환값의 log변환된 값은 정규 분포 형태를 가지는 검색 공간 설정 ( 값이 너무 크면 쓴다. )
                            
# def objective_func(search_space):
#     x1 = search_space['x1']
#     x2 = search_space['x2']
#     return_value =  x1**2 - 20*x2 
    
#     return return_value

# trials_val = Trials()

# best = fmin(
#     fn = objective_func,
#     space=search_space,
#     algo=tpe.suggest, # 알고리즘, 디폴트
#     max_evals=20,   # 서치 횟수
#     trials=trials_val,
#     rstate=np.random.default_rng(seed=10),
#     # rstate=333, # 차이가 뭘까?
# )

# print(best)
# print(trials_val.results)
# # {'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}...
# print(trials_val.vals)
# # {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0], 
# # 'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}
# # 판다스 데이터 프레임 사용

# results_df = pd.DataFrame()
# for i, result in enumerate(trials_val.results):
#     df_temp = pd.DataFrame(result, index=[i])
#     df_temp['iter'] = i
#     results_df = pd.concat([results_df, df_temp], axis=0, ignore_index=True)

# vals_df = pd.DataFrame(trials_val.vals)
# vals_df['iter'] = vals_df.index

# results_df['target'] = results_df['loss']

# print("Results DataFrame:")
# print(results_df.set_index(['iter', 'target', 'x1', 'x2']))

# print("\nVals DataFrame:")
# print(vals_df.set_index(['iter']))

# import numpy as np
# import hyperopt
# import pandas as pd
# from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# search_space = {'x1': hp.quniform('x1', -10, 10, 1), 'x2': hp.quniform('x2', -15, 15, 1)}

# def objective_func(search_space):
#     x1 = search_space['x1']
#     x2 = search_space['x2']
#     return x1 ** 2 - 20 * x2

# trials_val = Trials()

# best = fmin(
#     fn=objective_func,
#     space=search_space,
#     algo=tpe.suggest,
#     max_evals=20,
#     trials=trials_val,
#     rstate=np.random.default_rng(seed=10),
# )

# results_df = pd.DataFrame()
# for i, result in enumerate(trials_val.results):
#     df_temp = pd.DataFrame(result, index=[i])
#     df_temp['iter'] = i
#     results_df = pd.concat([results_df, df_temp], axis=0, ignore_index=True)

# vals_df = pd.DataFrame(trials_val.vals)
# vals_df['iter'] = vals_df.index

# results_df['target'] = results_df['loss']
# results_df['x1'] = vals_df['x1']
# results_df['x2'] = vals_df['x2']

# print("Results DataFrame:")
# print(results_df.set_index(['iter', 'target', 'x1', 'x2']))

# # print("\nVals DataFrame:")
# # print(vals_df.set_index(['iter']))

import numpy as np
import hyperopt as hp
from hyperopt import *
# print(hy.__version__)   # 0.2.7

search_space = {
    'x1': hp.quniform('x1', -10, 10, 1),
    'x2': hp.quniform('x2', -15, 15, 1),}

'''
hp.quniform(label, low, high, q) : label에 대해 low부터 high까지 q 간격으로 검색공간설정
hp.uniform(label, low, high): low부터 high까지 정규분포 형태로 검색공간설정
hp.randint(label, upper): 0부터 upper까지 random한 정수값으로 검색공간설정
hp.loguniform(label low, high) exp(uniform(low,high))값을 반환하며,
    반환값의 log변환된 값은 정규분포 형태를 가지는 검색공간 설정
'''
def objective_function(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_val =  x1**2 - 20*x2
    
    return return_val

trial_val = Trials()

best = fmin(
    fn= objective_function, # 목적함수
    space= search_space,    # 탐색범위
    algo= tpe.suggest,      # 알고리즘, default
    max_evals= 20,          # 탐색횟수
    trials= trial_val,      
    rstate= np.random.default_rng(seed=10)  # random state
)

print(best)

print(trial_val.results)
print(trial_val.vals)

'''
print('|   iter   |  target  |    x1    |    x2    |')
print('---------------------------------------------')
x1_list = trial_val.vals['x1']
x2_list = trial_val.vals['x2']
for idx, data in enumerate(trial_val.results):
    loss = data['loss']
    print(f'|{idx:^10}|{loss:^10}|{x1_list[idx]:^10}|{x2_list[idx]:^10}|')
'''

import pandas as pd
target = [aaa['loss'] for aaa in trial_val.results]
print(target)
df = pd.DataFrame({'target' : target,
                     'x1' : trial_val.vals['x1'],
                     'x2' : trial_val.vals['x2'],
                   })
print(df)

