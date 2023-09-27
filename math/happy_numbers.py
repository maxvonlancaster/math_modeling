import time

# unhappy_list = [20, 89, 37, 16, 4, 42, 58, 145]
# take natural number, sum the squers of its digits, get new number - go on. 
# sooner or later you either get 1 (happy number) or 42 (unhappy number)
unhap = []
hap = []

def get_squeres(i):
    res = 0
    splits = list(str(i))
    for s in splits:
        res += int(s)* int(s)
    return res

def calculate_squeres(i):
    list_unique = []
    a = i
    first = get_squeres(i)
    while True:
        list_unique.append(a)
        a = get_squeres(a)
        # if a in list_unique:
        if (a == 42) or (a in unhap):
            a = 42
            if(first not in unhap):
                unhap.append(first)
            break
        if (a == 1) or (a in hap):
            a = 1
            if(first not in hap):
                hap.append(first)
            break
    return a

start = time.time()
for i in range(1994, 2030):
    res = calculate_squeres(i)
    print(f'{i} : {res} \n')

end = time.time()
elapsed = end - start
print(f'elapsed time: {elapsed} seconds')
