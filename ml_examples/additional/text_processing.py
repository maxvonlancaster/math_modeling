import math

# def word_count(str):
#     counts = dict()
#     words = str.split()

#     for word in words:
#         if word in counts:
#             counts[word] += 1
#         else:
#             counts[word] = 1

#     return counts


# calculate the cosine between two vectors of the same dimension:
def _calculate_cosine(v1, v2):
    product, dv1, dv2 = 0, 0, 0
    for i in range(len(v1)):
        product += v1[i]*v2[i]
        dv1 += v1[i]*v1[i]
        dv2 += v2[i]*v2[i]
    return product / (dv1 * dv2)
        
# text input
input = [
    "the goal of this lecture is to explain the basics of free text processing",
    "the bag of words model is one such approach",
    "bird bird bird hello world"]

# split each text into array of words
splits = []
for text in input:
    splits.append(text.split(" "))

# create matrix for each unique word across texts with numbers representing how many times each word is 
# encountered in every text
unique, matrix = [], []
dict_count, dict_inverse_coef = dict(), dict()
for text_array in splits:
    for word in text_array:
        if not (word in unique):
            unique.append(word)

i = 0
for word in unique:
    a = []
    dict_count[i] = word
    for text_split in splits:
        a.append(text_split.count(word))
    matrix.append(a)
    i += 1

# represent each text as vectors
vectors = []
i = 0
for text in input:
    vector = []
    for column in matrix:
        vector.append(column[i])
    vectors.append(vector)
    i += 1

# inverse coeficcient calculation - to get rid of influence of common words
for i in range(len(matrix)):
    count = 0
    for elem in matrix[i]:
        if elem != 0:
            count += 1
    coef = math.log(len(matrix[i])/count)
    for j in range(len(matrix[i])):
        matrix[i][j] *= coef

print(matrix)

for i in range(len(input)-1):
    for j in range(i+1,len(input)):
        print(f"cosine between texts {i} and {j} is {_calculate_cosine(vectors[i], vectors[j])}")
