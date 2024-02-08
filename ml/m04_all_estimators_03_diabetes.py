from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터 

#1. 데이터 // 판다스, 넘파이 
path = "C:\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)    
test_csv = pd.read_csv(path + "test.csv", index_col=0)      # 헤더는 기본 첫번째 줄이 디폴트값
submission_csv = pd.read_csv(path + "sample_submission.csv")       

############# x 와 y를 분리 ################
x = train_csv.drop(['Outcome', 'Insulin'], axis=1)   # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제
y = train_csv.drop(['Insulin'], axis=1)       # 행삭제 : axis = 0 // 열삭제 : axis = 1 // train_csv에 있는 'Outcome'열 삭제 
y = train_csv['Outcome']                      # train_csv에 있는 'Outcome'열을 y로 설정
test_csv = test_csv.drop(['Insulin'], axis=1)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.9, random_state=123123,
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
allAlgorithms = all_estimators(type_filter='classifier')
for name, algorithm in allAlgorithms:
    try:
        #2. 모델
        model = algorithm()
        #.3 훈련
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)   
        print(name, "의 정답률 : ", round(acc, 2))   
    except:
        # print(name, '은 바보 멍충이!!!')  
        continue    #그냥 다음껄로 넘어간다.
    
'''
#.3 컴파일 훈련 // 하나 포기하는게 과적합 안걸림 // 
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("model.score : ", results)  


# LinearSVC                     0.6060606060606061
# Perceptron                  0.7121212121212122
# LogisticRegression          0.7878787878787878
# KNeighborsClassifier        0.7272727272727273
# DecisionTreeClassifier      0.7727272727272727
# RandomForestClassifier        0.803030303030303
'''
'''
daBoostClassifier 의 정답률 :  0.79
BaggingClassifier 의 정답률 :  0.77
BernoulliNB 의 정답률 :  0.68
CalibratedClassifierCV 의 정답률 :  0.8
DecisionTreeClassifier 의 정답률 :  0.82
DummyClassifier 의 정답률 :  0.61
ExtraTreeClassifier 의 정답률 :  0.71
ExtraTreesClassifier 의 정답률 :  0.83
GaussianNB 의 정답률 :  0.76
GaussianProcessClassifier 의 정답률 :  0.8
GradientBoostingClassifier 의 정답률 :  0.77
HistGradientBoostingClassifier 의 정답률 :  0.85
KNeighborsClassifier 의 정답률 :  0.8
LabelPropagation 의 정답률 :  0.71
LabelSpreading 의 정답률 :  0.71
LinearDiscriminantAnalysis 의 정답률 :  0.79
LinearSVC 의 정답률 :  0.8
LogisticRegression 의 정답률 :  0.79
LogisticRegressionCV 의 정답률 :  0.79
MLPClassifier 의 정답률 :  0.83
NearestCentroid 의 정답률 :  0.67
NuSVC 의 정답률 :  0.83
PassiveAggressiveClassifier 의 정답률 :  0.8
Perceptron 의 정답률 :  0.71
QuadraticDiscriminantAnalysis 의 정답률 :  0.74
RandomForestClassifier 의 정답률 :  0.77
RidgeClassifier 의 정답률 :  0.8
RidgeClassifierCV 의 정답률 :  0.8
SGDClassifier 의 정답률 :  0.76
SVC 의 정답률 :  0.82
'''


