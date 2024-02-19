import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])   # -10 과 50은 이상치
print(aaa.shape)    # (13,)
aaa = aaa.reshape(-1, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3) # contamination 비율에 따라 데이터의 일정 부분을 이상치로 간주

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)  # [-1  1  1  1  1  1  1  1  1  1  1  1 -1] // 



