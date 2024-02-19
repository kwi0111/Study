import numpy as np
aaa = np.array([[-10,2,3,4,5,
                 6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,
                600,-70000,800,900,1000,210,420,350]]).T    # 2열을 1개로 인식 // 
print(aaa.shape)    # (13, 2)


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1) # contamination 비율에 따라 데이터의 일정 부분을 이상치로 간주

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)  