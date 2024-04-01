import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#1. 데이터
path = 'C:\\_data\\kaggle\\Obesity_Risk\\'
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
class_label = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad']

''' # train csv 라벨들 확인
for label in train_csv:
        if label in class_label:
                print(label ,np.unique(train_csv[label], return_counts=True))
Gender (array(['Female', 'Male'], dtype=object), array([10422, 10336], dtype=int64))
family_history_with_overweight (array(['no', 'yes'], dtype=object), array([ 3744, 17014], dtype=int64))
FAVC (array(['no', 'yes'], dtype=object), array([ 1776, 18982], dtype=int64))
CAEC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  478,  2472, 17529,   279], dtype=int64))
SMOKE (array(['no', 'yes'], dtype=object), array([20513,   245], dtype=int64))
SCC (array(['no', 'yes'], dtype=object), array([20071,   687], dtype=int64))
CALC (array(['Frequently', 'Sometimes', 'no'], dtype=object), array([  529, 15066,  5163], dtype=int64))
MTRANS (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation',
       'Walking'], dtype=object), array([ 3534,    32,    38, 16687,   467], dtype=int64))
NObeyesdad (array(['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II'], dtype=object), array([2523, 3082, 2910, 3248, 4046, 2427, 2522], dtype=int64))
'''
''' # test csv 라벨들 확인
for label in test_csv:
        if label in class_label:
                print(label, np.unique(test_csv[label], return_counts=True))
Gender (array(['Female', 'Male'], dtype=object), array([6965, 6875], dtype=int64))
family_history_with_overweight (array(['no', 'yes'], dtype=object), array([ 2456, 11384], dtype=int64))
FAVC (array(['no', 'yes'], dtype=object), array([ 1257, 12583], dtype=int64))
CAEC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  359,  1617, 11689,   175], dtype=int64))
SMOKE (array(['no', 'yes'], dtype=object), array([13660,   180], dtype=int64))
SCC (array(['no', 'yes'], dtype=object), array([13376,   464], dtype=int64))
CALC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([   2,  346, 9979, 3513], dtype=int64))
MTRANS (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation',
       'Walking'], dtype=object), array([ 2405,    25,    19, 11111,   280], dtype=int64))
'''
test_csv.loc[test_csv['CALC'] == 'Always', 'CALC'] = 'Frequently'
# print(train_csv.head)
''' # train head 확인
<bound method NDFrame.head of        
        Gender      Age    Height      Weight   family_history_with_overweight FAVC      FCVC     NCP        CAEC   SMOKE    CH2O   SCC    FAF       TUE       CALC                 MTRANS           NObeyesdad
id
0        Male  24.443011  1.699998   81.669950                            yes  yes  2.000000  2.983297   Sometimes    no  2.763573  no  0.000000  0.976473  Sometimes  Public_Transportation  Overweight_Level_II
1      Female  18.000000  1.560000   57.000000                            yes  yes  2.000000  3.000000  Frequently    no  2.000000  no  1.000000  1.000000         no             Automobile        Normal_Weight
2      Female  18.000000  1.711460   50.165754                            yes  yes  1.880534  1.411685   Sometimes    no  1.910378  no  0.866045  1.673584         no  Public_Transportation  Insufficient_Weight
3      Female  20.952737  1.710730  131.274851                            yes  yes  3.000000  3.000000   Sometimes    no  1.674061  no  1.467863  0.780199  Sometimes  Public_Transportation     Obesity_Type_III
4        Male  31.641081  1.914186   93.798055                            yes  yes  2.679664  1.971472   Sometimes    no  1.979848  no  1.967973  0.931721  Sometimes  Public_Transportation  Overweight_Level_II
...       ...        ...       ...         ...                            ...  ...       ...       ...         ...   ...       ...  ..       ...       ...        ...                    ...                  ...
20753    Male  25.137087  1.766626  114.187096                            yes  yes  2.919584  3.000000   Sometimes    no  2.151809  no  1.330519  0.196680  Sometimes  Public_Transportation      Obesity_Type_II
20754    Male  18.000000  1.710000   50.000000                             no  yes  3.000000  4.000000  Frequently    no  1.000000  no  2.000000  1.000000  Sometimes  Public_Transportation  Insufficient_Weight
20755    Male  20.101026  1.819557  105.580491                            yes  yes  2.407817  3.000000   Sometimes    no  2.000000  no  1.158040  1.198439         no  Public_Transportation      Obesity_Type_II
20756    Male  33.852953  1.700000   83.520113                            yes  yes  2.671238  1.971472   Sometimes    no  2.144838  no  0.000000  0.973834         no             Automobile  Overweight_Level_II
20757    Male  26.680376  1.816547  118.134898                            yes  yes  3.000000  3.000000   Sometimes    no  2.003563  no  0.684487  0.713823  Sometimes  Public_Transportation      Obesity_Type_II
'''
# print(train_csv.columns)
# ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#        'CALC', 'MTRANS', 'NObeyesdad']
x_labelEncoder = LabelEncoder()
train_csv['Gender'] = x_labelEncoder.fit_transform(train_csv['Gender'])
train_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(train_csv['family_history_with_overweight'])
train_csv['FAVC'] = x_labelEncoder.fit_transform(train_csv['FAVC'])
train_csv['CAEC'] = x_labelEncoder.fit_transform(train_csv['CAEC'])
train_csv['SMOKE'] = x_labelEncoder.fit_transform(train_csv['SMOKE'])
train_csv['SCC'] = x_labelEncoder.fit_transform(train_csv['SCC'])
train_csv['CALC'] = x_labelEncoder.fit_transform(train_csv['CALC'])
train_csv['MTRANS'] = x_labelEncoder.fit_transform(train_csv['MTRANS'])

x_labelEncoder = LabelEncoder()
test_csv['Gender'] = x_labelEncoder.fit_transform(test_csv['Gender'])
test_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(test_csv['family_history_with_overweight'])
test_csv['FAVC'] = x_labelEncoder.fit_transform(test_csv['FAVC'])
test_csv['CAEC'] = x_labelEncoder.fit_transform(test_csv['CAEC'])
test_csv['SMOKE'] = x_labelEncoder.fit_transform(test_csv['SMOKE'])
test_csv['SCC'] = x_labelEncoder.fit_transform(test_csv['SCC'])
test_csv['CALC'] = x_labelEncoder.fit_transform(test_csv['CALC'])
test_csv['MTRANS'] = x_labelEncoder.fit_transform(test_csv['MTRANS'])

y_labelEncoder = LabelEncoder()
train_csv['NObeyesdad'] = y_labelEncoder.fit_transform(train_csv['NObeyesdad'])
# print(train_csv.head)
''' # 라벨 인코딩 후 train.head 확인
<bound method NDFrame.head of        
       Gender     Age      Height     Weight   family_history_with_overweight    FAVC    CVC       NCP     CAEC  SMOKE   CH2O    SCC    FAF       TUE      CALC   MTRANS    NObeyesdad
id
0           1  24.443011  1.699998   81.669950                               1     1  2.000000  2.983297     2      0  2.763573    0  0.000000  0.976473     1       3           6
1           0  18.000000  1.560000   57.000000                               1     1  2.000000  3.000000     1      0  2.000000    0  1.000000  1.000000     2       0           1
2           0  18.000000  1.711460   50.165754                               1     1  1.880534  1.411685     2      0  1.910378    0  0.866045  1.673584     2       3           0
3           0  20.952737  1.710730  131.274851                               1     1  3.000000  3.000000     2      0  1.674061    0  1.467863  0.780199     1       3           4
4           1  31.641081  1.914186   93.798055                               1     1  2.679664  1.971472     2      0  1.979848    0  1.967973  0.931721     1       3           6
...       ...        ...       ...         ...                             ...   ...       ...       ...   ...    ...       ...  ...       ...       ...   ...     ...         ...
20753       1  25.137087  1.766626  114.187096                               1     1  2.919584  3.000000     2      0  2.151809    0  1.330519  0.196680     1       3           3
20754       1  18.000000  1.710000   50.000000                               0     1  3.000000  4.000000     1      0  1.000000    0  2.000000  1.000000     1       3           0
20755       1  20.101026  1.819557  105.580491                               1     1  2.407817  3.000000     2      0  2.000000    0  1.158040  1.198439     2       3           3
20756       1  33.852953  1.700000   83.520113                               1     1  2.671238  1.971472     2      0  2.144838    0  0.000000  0.973834     2       0           6
20757       1  26.680376  1.816547  118.134898                               1     1  3.000000  3.000000     2      0  2.003563    0  0.684487  0.713823     1       3           3
'''


""" # P 검정
import scipy.stats as stats
for label in train_csv:
    print(label," ",stats.pearsonr(train_csv['NObeyesdad'],train_csv[label]))
Gender   PearsonRResult(statistic=0.046574912033978184, pvalue=1.8990220650683642e-11)
Age   PearsonRResult(statistic=0.2830183712239907, pvalue=0.0)
Height   PearsonRResult(statistic=0.060785550400480136, pvalue=1.8621467518362792e-18)
Weight   PearsonRResult(statistic=0.43182097207728903, pvalue=0.0)
family_history_with_overweight   PearsonRResult(statistic=0.32132484319938587, pvalue=0.0)
        FAVC   PearsonRResult(statistic=0.010176246176913503, pvalue=0.14261934009881228)
FCVC   PearsonRResult(statistic=0.0410763864808035, pvalue=3.2139644150763383e-09)
NCP   PearsonRResult(statistic=-0.09115416942846906, pvalue=1.4953264119959517e-39)
CAEC   PearsonRResult(statistic=0.297419757072015, pvalue=0.0)
        SMOKE   PearsonRResult(statistic=-0.0013927633524938529, pvalue=0.84097046043243)
CH2O   PearsonRResult(statistic=0.18709958001025073, pvalue=7.344610346043163e-163)
SCC   PearsonRResult(statistic=-0.06517135385355988, pvalue=5.504265380370881e-21)
FAF   PearsonRResult(statistic=-0.09664292513984239, pvalue=2.9003365419591887e-44)
TUE   PearsonRResult(statistic=-0.07603955730067295, pvalue=5.284478962295593e-28)
CALC   PearsonRResult(statistic=-0.16849742485140495, pvalue=5.0277297570113e-132)
MTRANS   PearsonRResult(statistic=-0.07743006081693204, pvalue=5.594245339542871e-29)
NObeyesdad   PearsonRResult(statistic=1.0, pvalue=0.0) """

''' BMI 컬럼 추가 '''
train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height']*train_csv['Height'])
test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height']*test_csv['Height'])

''' 이상치 제거 '''
age_q1 = train_csv['Age'].quantile(0.25)
age_q3 = train_csv['Age'].quantile(0.75)
age_gap = (age_q3 - age_q1 ) * 1.5
age_under = age_q1 - age_gap
age_upper = age_q3 + age_gap
train_csv = train_csv[train_csv['Age']>=age_under]
train_csv = train_csv[train_csv['Age']<=age_upper]

weight_q1 = train_csv['Weight'].quantile(0.25)
weight_q3 = train_csv['Weight'].quantile(0.75)
weight_gap = (weight_q3 - weight_q1 ) * 1.5
weight_under = weight_q1 - weight_gap
weight_upper = weight_q3 + weight_gap
train_csv = train_csv[train_csv['Weight']>=weight_under]
train_csv = train_csv[train_csv['Weight']<=weight_upper]

x = train_csv.drop(['NObeyesdad'], axis=1) # P검정에 의거하여 FAVC와 SMOKE 제거
y = train_csv['NObeyesdad']

'''# 최대 최소 1분위 3분위 구하기
for label in x:         
        if label in class_label:
                continue
        print(f"{label:30}: max={max(x[label]):<10}  min={min(x[label]):<10}  q1={x[label].quantile(0.25):<10}  q3={x[label].quantile(0.75):<10}")
Age                           : max=61.0        min=14.0        q1=20.0        q3=26.0
Height                        : max=1.975663    min=1.45        q1=1.631856    q3=1.762887
Weight                        : max=165.057269  min=39.0        q1=66.0        q3=111.600553
FCVC                          : max=3.0         min=1.0         q1=2.0         q3=3.0
NCP                           : max=4.0         min=1.0         q1=3.0         q3=3.0
CH2O                          : max=3.0         min=1.0         q1=1.792022    q3=2.549617
FAF                           : max=3.0         min=0.0         q1=0.008013    q3=1.587406
TUE                           : max=2.0         min=0.0         q1=0.0         q3=1.0
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
y = y.reshape(-1,1)
y = OneHotEncoder(sparse=False).fit_transform(y)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size= 0.8, stratify=y)
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
scaler = MinMaxScaler() # 클래스 정의
# scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = np.float32(x_train)
x_test = np.float32(x_test)
y_train = np.float32(y_train)
y_test = np.float32(y_test)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 17])
w = tf.compat.v1.Variable(tf.random_normal([17, 7]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1, 7]), name = 'bias')
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(xp, w) + b)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(hypothesis), axis = 1)) 

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-1)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1000
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict = {xp:x_train, yp:y_train})
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)


y_predict = sess.run(hypothesis, feed_dict = {xp:x_test})
print(y_predict)   
y_predict = sess.run(tf.argmax(y_predict, 1))
print(y_predict) 
y_data_arg = np.argmax(y_test, 1)
print(y_data_arg)    

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data_arg)
print('ACC : ', acc) 
# ACC :  0.34913217623498
sess.close()





















