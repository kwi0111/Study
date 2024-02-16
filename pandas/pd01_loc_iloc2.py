# 판다스 
import pandas as pd

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
    
    
]

index = ["031", "059", "033", "045", "023"] # str // 인덱스는 연산때 쓰는 데이터 아니다.
columns = ["종목명", "시가", "종가"]

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
print('===================== 시가가 1100원 이상인 행을 모두 뽑아라 ==========================')
# selected_rows = []
# for index, row in df.iterrows():
#     if int(row["시가"]) >= 1100:
#         selected_rows.append(row)
        
# selected_df = pd.DataFrame(selected_rows, columns=columns)

# print("\n1100원 이상인 행:\n", selected_df)
# 시가가 1100원 이상인 행을 loc를 사용하여 선택

# selected_df = df.loc[df["시가"].astype(int) >= 1100]
# print("\n1100원 이상인 행:\n", selected_df)

# selected_df = df.iloc[[i for i, price in enumerate(df["시가"]) if int(price) >= 1100]]
# print("\n1100원 이상인 행:\n", selected_df)
'''
aaa = df["시가"] >= '1100'
print(aaa)
# 031    False
# 059     True
# 033     True
# 045     True
# 023    False
print(df[aaa])  # true인 부분만 출력
print(df.loc[aaa])
# print(df.iloc[aaa]) # 에러
print(df[df['시가'] >= '1100']) # 이 표현이 제일 많다.
'''

print('===================== 시가가 1100원 이상인 종가만 뽑아라 ==========================')
# selected_df = df.loc[df["시가"].astype(int) >= 1100]
# print(df.loc[df["시가"].astype(int) >= 1100, ["종목명", "종가"]])
# print("************")
# print(selected_df[["종목명", "종가"]])
print(df[df['시가'] >= '1100']["종가"])
# print(df[df['시가'] >= '1100'][2]) # 에러
print(df.loc[df['시가'] >= '1100']["종가"])
print(df.loc[df['시가'] >= '1100' ,"종가"])

