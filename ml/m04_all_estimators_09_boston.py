# 보스턴에 관한 데이터
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
datasets = load_boston()        # 변수에 집어 넣은다음 프린트
print(datasets)
x = datasets.data           # x에서 스케일링
y = datasets.target         # y 건들지 않는다.

x_train, x_test, y_train, y_test = train_test_split(x, y,               
                                                    train_size=0.7,
                                                    random_state=1140,     
                                                    shuffle=True,
                                                    )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성 
allAlgorithms = all_estimators(type_filter='regressor')

for name, algorithm in allAlgorithms:
    try:
        #2. 모델
        model = algorithm()
        #.3 훈련
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)   
        print(name, "의 정답률 : ", round(acc, 2))   
    except: 
        continue
'''
#3. 컴파일, 훈련 
start_time = time.time()   #현재 시간
model.fit(x_train, y_train)
end_time = time.time()   #끝나는 시간

#4. 평가, 예측
results = model.score(x_test, y_test) 
y_predict = model.predict(x_test) 
print("acc : ", results)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# acc :  0.5275130859946872
# 걸린시간 :  0.01 초

# 스케일러 x
# acc :  0.19233716312185622
# 걸린시간 :  0.01 초

# LinearSVR                   0.5198578310569075  
# LinearRegression            0.739750714020937
# KNeighborsRegressor         0.5389429273211952
# DecisionTreeRegressor       0.8418772268815504
# RandomForestRegressor        0.9210620914378058
'''

'''
ARDRegression 의 정답률 :  0.74
AdaBoostRegressor 의 정답률 :  0.88
BaggingRegressor 의 정답률 :  0.9
BayesianRidge 의 정답률 :  0.74
DecisionTreeRegressor 의 정답률 :  0.85
DummyRegressor 의 정답률 :  -0.0
ElasticNet 의 정답률 :  0.66
ElasticNetCV 의 정답률 :  0.74
ExtraTreeRegressor 의 정답률 :  0.77
ExtraTreesRegressor 의 정답률 :  0.92
GammaRegressor 의 정답률 :  0.68
GaussianProcessRegressor 의 정답률 :  0.29
GradientBoostingRegressor 의 정답률 :  0.93
HistGradientBoostingRegressor 의 정답률 :  0.91
HuberRegressor 의 정답률 :  0.71
KNeighborsRegressor 의 정답률 :  0.76
KernelRidge 의 정답률 :  -5.13
Lars 의 정답률 :  0.7
LarsCV 의 정답률 :  0.72
Lasso 의 정답률 :  0.67
LassoCV 의 정답률 :  0.74
LassoLars 의 정답률 :  -0.0
LassoLarsCV 의 정답률 :  0.74
LassoLarsIC 의 정답률 :  0.74
LinearRegression 의 정답률 :  0.74
LinearSVR 의 정답률 :  0.7
MLPRegressor 의 정답률 :  0.71
NuSVR 의 정답률 :  0.62
OrthogonalMatchingPursuit 의 정답률 :  0.54
OrthogonalMatchingPursuitCV 의 정답률 :  0.69
PLSRegression 의 정답률 :  0.72
PassiveAggressiveRegressor 의 정답률 :  0.63
PoissonRegressor 의 정답률 :  0.8
QuantileRegressor 의 정답률 :  -0.03
RANSACRegressor 의 정답률 :  0.63
RandomForestRegressor 의 정답률 :  0.92
Ridge 의 정답률 :  0.74
RidgeCV 의 정답률 :  0.74
SGDRegressor 의 정답률 :  0.74
SVR 의 정답률 :  0.62
TheilSenRegressor 의 정답률 :  0.62
TransformedTargetRegressor 의 정답률 :  0.74
TweedieRegressor 의 정답률 :  0.65

'''


