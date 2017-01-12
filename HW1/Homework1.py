import numpy


"""Step 1: Simulate Rankings of Relevance for E and P (5 points)
In the first step you will generate pairs of rankings of relevance, for the production P and experimental E, respectively, for a hypothetical query q. Assume a 3-graded relevance, i.e. {N, R, HR}. Construct all possible P and E ranking pairs of length 5, for which E outperforms P. <= (10/1, 11:06: Postpone the discarding to Step 3)

Example:
P: {N N N N N}
E: {N N N N R}
...
P: {HR HR HR HR R}
E: {HR HR HR HR HR}

(Note 1: If you do not have enough computational power, sample 1000 pair uniformly at random to show your work.)"""


import itertools
import numpy

def split_list(a_list):
    half = len(a_list) / 2
    return [a_list[:half], a_list[half:]]
sum = 0
lijst=[]
for i in itertools.product(["N", "R", "HR"], repeat=10):
    sum += 1
    i = list(i)
    i = split_list(i)
    lijst.append(i)


'''
Calculates precision at rank k with a list with 3 relevance levels (R, HR and N).
'Precision at rank k' though, asks for a binary classication problem,
so HR and R is counted as relevant (1) and N as non-relevant(0).

'''
def precisionAtK(k, lijst):
    countTP = 0 # amount of true positives
    countFP = 0 # amount of false positives
    precisionList =[]

    for j in lijst:
        kcounter = 0
        for m in range(0, k-1):
            l = j[0][m]
            if l == 'R': countTP+=1
            elif l == 'HR': countTP+=1
            else : countFP+=1
            precisionP = countTP/float(countTP+countFP)
        countTP=0
        countFP=0
        for m in range(0,k-1):
            l = j[1][m]
            if l == 'R': countTP+=1
            elif l == 'HR': countTP+=1
            else : countFP+=1
            precisionE = countTP / float(countTP + countFP)
        precisions = [precisionP, precisionE]
        precisionList.append(precisions)
    print precisionList
precisionAtK(5, lijst)
#for i in length(lijst):

#precision = TP/(TP+FP)

#def precisionAtK(combinations):
  #  for length(combinations):
     #   combination = combinations[i]
     #   combination =
#

