import pandas as pd

data = [
    ["삼성","1000","2000"],
    ["현대","1100","3000"],
    ["LG","2000","500"],
    ["아모레","3500","6000"],
    ["네이버","100","1500"],
]

index = ["031", "059", "033", "045","023"]
columns = ["종목명","시가","종가"]

datasets = pd.DataFrame(data, index=index, columns=columns)
print(datasets)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

print("===== 시가 1100원 이상 뽑기 (행단위로 출력) =====")
print(datasets[datasets["시가"].astype(int) >= 1100])
print(datasets.loc[datasets["시가"].astype(int) >= 1100])   # 위와 같음
#      종목명    시가    종가
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
print(datasets[datasets["시가"] >= '1100']) # 결과는 같으나 위험함 파이썬의 문자열 대소 비교는 조심해야함
print("===== 시가 1100원 이상 뽑기 (종가만 출력) =====")
print(datasets[datasets["시가"].astype(int) >= 1100]['종가'])
# 059    3000
# 033     500
# 045    6000
# Name: 종가, dtype: object
print([datasets.iloc[idx]['종가'] for idx, result in enumerate(datasets['시가']) if int(result) >= 1100])
# ['3000', '500', '6000']
print([idx for idx in datasets['시가'].index])  # ['031', '059', '033', '045', '023']

print('30' < '100') # false