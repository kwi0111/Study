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
print('===============================================')
# print(df[0]) #에러
# print(df["031"])   #에러
print(df["종목명"])   # 판다스에서는 기준 : 컬럼(열) // 항상 컬럼명으로 찍는다.
######## 아모레 출력하고 싶다. ########
# print(df[4, 0])    # 에러
# print(df["종목명", "045"])    # key에러
print(df["종목명"]["045"])    # 아모레
######## 판다스는 열 -> 행 순서대로 찍는다.

# loc : 인덱스를 기준으로 행 데이터 추출
# iloc : 행번호를 기준으로 행 데이터 추출
        # 인트 loc 
print('=================== 아모레 뽑자 ========================')
print(df.loc["045"]) # 아모레의 행 // 
print('======================================================')
# print(df.loc[3]) # key 에러
print('======================================================')
print(df.iloc[3])   # 아모레 행 // 
print('==================== 네이버 뽑자 ============================')
print(df.loc["023"])   # 네이버 행 // 
print(df.iloc[-1])   # -1이나 4로 출력 
print(df.iloc[4])
print('==================== 아모레 시가 (3500) ============================')
print(df.loc["045"].loc["시가"])    # 이게 가장 가독성 좋다.
print(df.loc["045"].iloc[1])
print(df.iloc[3].iloc[1])
print(df.iloc[3].loc["시가"])

print(df.loc["045"][1])
print(df.iloc[3][1])

print(df.loc["045"]["시가"])
print(df.iloc[3]["시가"])
print(df.loc["045", "시가"])
print(df.iloc[3, 1])

print("+++++++++++++++++ 아모레와 네이버의 시가 뽑자 +++++++++++++++++++")
print(df.iloc[3:5, 1])
print(df.iloc[[3,4], 1])    # 행부분 리스트로 묶어준다.
# print(df.iloc[3:5, "시가"]) # 에러
# print(df.iloc[[3,4], "시가"]) # 에러

# print(df.loc[3:5, "시가"]) # 에러
# print(df.loc[["045","023"], 1])   # 에러 
print(df.loc["045":"023", "시가"])   # 돼
# print(df.loc["045","023", 1])   # 에러
print(df.loc[["045","023"], "시가"])    # 돼







