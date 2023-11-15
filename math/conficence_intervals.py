import numpy as np 
import scipy.stats as st 
import random
import csv
import os
import matplotlib.pyplot as plt
  

################################## PART 1


# define sample data 
gfg_data = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3,  
            3, 4, 4, 5, 5, 5, 6, 7, 8, 10] 
  
mean = np.mean(gfg_data)
standart_deviation = st.sem(gfg_data)
print(f'population mean: {mean}')
print(f'standart deviation: {standart_deviation}')

# create 90% confidence interval 
interval = st.t.interval(alpha=0.90, df=len(gfg_data), 
              loc=mean, 
              scale=standart_deviation) 

print(interval)


################################### PART 2


# mu, sigma = 5, 2 # mean and standard deviation
# random_list = np.random.normal(mu, sigma, 1000)
# print(random_list)

# interval_random = st.t.interval(alpha=0.90, df=len(random_list), 
#               loc=np.mean(random_list), 
#               scale=st.sem(random_list)) 
# print(interval_random)

# in_interval, out_interval = 0, 0
# for i in range(10000):
#     if(interval_random[0] < random_list[random.randint(1,999)] < interval_random[1]):
#         in_interval += 1
#     else:
#         out_interval += 1
# print(f'in the interval: {in_interval} outside: {out_interval}')

# mean_random = np.mean(random_list)
# standart_deviation_random = st.sem(random_list)

# count, bins, ignored = plt.hist(random_list, 100)
# plt.show()


################################### PART 3


# script_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_dir, 'ratings.csv')
# ratings = []

# with open(file_path, 'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     for row in csv_reader:
#         if 'Comedy' in row[9]:
#             ratings.append(int(row[1]))

# print(ratings)

# interval_ratings = st.t.interval(alpha=0.99, df=len(ratings), 
#               loc=np.mean(ratings), 
#               scale=st.sem(ratings)) 
# print(interval_ratings)


