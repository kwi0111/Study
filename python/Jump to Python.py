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

 




