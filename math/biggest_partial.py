import random

# given the list of numbers, get the biggest partial of two numbers in a list 
# (i.e. max(a/b), a,b from list);
# do not use len() method and for loops

# get random list
def get_list():
    list = []
    n = random.randint(5,20)
    for i in range(n):
        list.append(random.randint(-99,99))
    return list


# i hate this solution, but it is the best one i got so far
list = get_list()
print(list)

list_neg = []
list_pos = []
while list:
    if list[0] < 0:
        list_neg.append(list.pop(0))
    elif list[0] > 0:
        list_pos.append(list.pop(0))
    else:
        list.pop(0)

list_neg.sort()
list_pos.sort()

print(list_neg)
print(list_pos)
min_n = list_neg.pop(0)
min_p = list_pos.pop(0)
max_neg = 0
max_pos = 0

while list_neg:
    max_neg = min_n / list_neg.pop(0)

while list_pos:
    max_pos = list_pos.pop(0) / min_p

print(max_neg)
print(max_pos)
print(max(max_pos,max_neg))
