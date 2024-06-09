# from typing import Dict

# def generate_squares(num: int)-> Dict[int, int]:
#     if num==0:
#         return {}
#     else:
#         return {k:k**2 for k in range(1,num+1)}
    
# print(generate_squares(10))
# print(generate_squares(5))
# print(generate_squares(0))

x = [1, 2, 3]
y = x.copy()
x[1] = 0
print(y)