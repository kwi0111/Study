'''
a = "Life is too short, You need python" 
print(a[0:4])
# a[시작번호 : 끝 번호] 끝 번호에 해당하는 문자는 포함하지 않음
print(a[19:])
# 끝번호 생략 // 시작번호부터 끝까지 뽑아낸다.
print(a[:17])
# 시작번호 생략 // 처음부터 끝까지 
print(a[:])
# 다나옴
print(a[19:-7]) # You need

a = "I eat %d apples" %3 # 숫자
print(a)
a = "I eat %s apples" %"five" # 문자
print(a)


a = [1,2,3]
# print(a[2] + 'hi') # TypeError: unsupported operand type(s) for +: 'int' and 'str' 정수와 문자열 더할수 없다.
print(str(a[2]) + 'hi') # str은 정수나 실수를 문자열로 바꿔준다.
'''
# print("== Program Start")

# try:
#     a = 99/0            # error
#     print(f"99/0 : {a}")
# except:
#     print("== error!!  but, still alive")

# print("== Program End")

# menu = ["냉면", "볶음밥", "피자", "짜장면"]
# while 1:
#     order = input("[메뉴를 선택해 주세요 (1.냉면, 2.볶음밥, 3.피자, 4.짜장면 (숫자로 입력해 주세요)] : ")
#     try:
#         print(menu[int(order) - 1] + "을 선택하셨습니다.")
#     except ValueError:
#         print("숫자만 입력하세요")
#     except IndexError:
#         print("없는 메뉴입니다.")
#     else:
#         break
#     finally:
#         print("--------------------------")

#상속
class 인공지능:
    type = "IT"
    def coding(self):
        print(f"type is {self.type}")
        print("do studing")

class Python:
    def ptn(self):
        print("do python")

class 머신러닝(인공지능, Python):
    type = "AI"
    def coding(self):
        print(f"type is {self.type}")
        print("do preprocessing")
    
class 딥러닝(머신러닝):
    def coding(self):
        print(f"type is {self.type}")
        print("do modeling")
AI = 인공지능()
ML = 머신러닝()
NN = 딥러닝()
AI.coding()
# type is IT
# do studing
ML.coding()
# type is AI
# do preprocessing
ML.ptn()
# do python
NN.coding()
# type is AI
# do modeling







