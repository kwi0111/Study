import numpy as np

aaa = np.array([[-10,2,3,4,5,
                 6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,
                600,-70000,800,900,1000,210,420,350]]).T  

# 판다스 = 시리즈 : 백터 (인덱스가 있다) / 데이터 프레임 : 행렬

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75]) # percentile(백분위수) : 데이터의 25% ㅈ, 50%, 75% 분위수를 계산
    print("1사분위 :", quartile_1)
    print("q2 :", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1   # IQR = q3 - q1 (사분위 범위)
    print("iqr : ", iqr)    
    lower_bound = quartile_1 - (iqr * 1.5) # 이 범위까지 정상 데이터로 판단하자. 1.5를 커스터마이징 가능 
    upper_bound = quartile_3 + (iqr * 1.5)  # 이 범위까지 정상 데이터로 판단하자.
    # lower_bound와 upper_bound는 이상치의 범위를 정의하는데, 일반적으로 사분위 범위를 기준으로 1.5배의 값을 더하거나 빼서 계산
    return np.where((data_out>upper_bound) |    # np.where = 이상치의 위치 인덱스 반환 // 19보다 큰놈 // |은 or를 의미
                     (data_out<lower_bound))    # -5보다 작은놈
    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc) 

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

#test 2
##### 과제
'''
pd04_1_따릉이.py
pd04_2_kaggle_bike.py
pd04_3_대출.py
pd04_04_kaggle_비만.py
'''






