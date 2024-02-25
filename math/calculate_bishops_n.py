def get_n(n):
    return 1 + (-10) * (n+1) + (29/2)*(n+2)*(n+1) - (16/3) * (n+3)*(n+2)*(n+1)+(1/2) * (n+4)*(n+3)*(n+2)*(n+1)

for i in range(50):
    print(f"For {i} : {round(get_n(i))}" )