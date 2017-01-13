##### STEP 1 #####
import itertools
import math
import random


nr_documents = 5

relevances_cats = ['N', 'R', 'HR']
relevances_vals = [0,1,5]
     
    
def split_list(a_list):
    half = len(a_list)/2
    return [a_list[:half], a_list[half:]]

def get_combinations_list(relevances):  
    combinations = []
    for i in itertools.product(relevances, repeat = nr_documents*2):
        i = list(i)
        i = split_list(i)
        combinations.append(i)
    return combinations

combinations_cats = get_combinations_list(relevances_cats)
combinations_vals = get_combinations_list(relevances_vals)
    
##### STEP 2 #####   

##### precision #####


'''
Calculates precision at rank k with a list with 3 relevance levels (R, HR and N).
'Precision at rank k' though, asks for a binary classication problem,
so HR and R is counted as relevant (1) and N as non-relevant(0).
k must be 5 or smaller
'''
def precisionAtK(k):
    countTP = 0 # amount of true positives
    countFP = 0 # amount of false positives
    precisionList =[]
    for j in combinations_cats:
        kcounter = 0
        for m in range(0, k):
            l = j[0][m]
            if l == 'R': countTP+=1
            elif l == 'HR': countTP+=1
            else : countFP+=1
            precisionP = countTP/float(countTP+countFP)
        countTP=0
        countFP=0
        for m in range(0,k):
            l = j[1][m]
            if l == 'R': countTP+=1
            elif l == 'HR': countTP+=1
            else : countFP+=1
            precisionE = countTP / float(countTP + countFP)
        precisions = [precisionP, precisionE]
        precisionList.append(precisions)
    return precisionList


##### dcg #####

'''
k must be 5 or smaller
'''
def dcg(k):
    EP_results = []
    for relevances in combinations_vals:
        rank_dcgs = []
        for algorithm in relevances:
            dcg = 0
            for r in range(1,k+1):
                dcg += ((2**algorithm[r-1])-1)/(math.log(1+r,2))
            rank_dcgs.append(dcg)
        EP_results.append(rank_dcgs)
    return EP_results

##### err #####

def R(seq, g): # mapping from relevance grades g to probability of relevance           
    return ((2**seq[g-1])-1)/float((2**max(relevances_vals)))

def P(seq, r): # probability that user stops at position r
    P = 1
    for i in range(1,(r-1)+1):
        P *= (1-R(seq, i)) * R(seq, r)
    return P 
                       
def err(): # a cascade based metric with x(r) = 1/r
    ERR_results = []
    for relevances in combinations_vals:
        rank_err = []
        for algorithm in relevances:
            err = 0
            for r in range(1, len(algorithm)+1):
                err += (1/float(r))*P(algorithm, r)
            rank_err.append(err)
        ERR_results.append(rank_err)
    return ERR_results 


##### STEP 3 #####

def calculateDMeasure(results):
    difference_measures=[]
    for algo in results:
        a = algo[0]
        b = algo[1]
        difference = b - a
#        print difference
        if (difference > 0 ): difference_measures.append(difference)
#    print difference_measures
    return difference_measures

#precision = precisionAtK(5)
#dcg = dcg(4)
#err = err()

#difference_measures = calculateDMeasure(precision)
#for difference in difference_measures:
#    print difference                       
    
##### STEP 4 #####

#Team-draft interleaving

def generated_clicks():
    clicks = []
    for i in range(nr_documents*2):
        clicks.append(random.getrandbits(1))
    return clicks

def flip_coin():
    random.seed()
    return random.getrandbits(1)
                       
def team_draft_interleaving(rankings, clicks):
    credits = [0,0]
    new_ranking = []
    for i in range(nr_documents):
        winner = flip_coin()
        
        new_ranking.append(rankings[winner][i])
        if clicks[len(new_ranking)-1] == 1:
            credits[winner] += 1
            
        new_ranking.append(rankings[1-winner][i])
        if clicks[len(new_ranking)-1] == 1:
            credits[1-winner] += 1
            
    return new_ranking, credits

rankings = [['N','R','HR','R','HR'], ['R','R','HR','N','N']]
clicks = generated_clicks()

### Team-draft interleaved ranking ###
team_draft_interleaved_ranking, credits = team_draft_interleaving(rankings, clicks)
#print 'Ranking P: ', rankings[0]
#print 'Ranking E: ', rankings[1]
#print 'Team-Draft Interleaved ranking: ', team_draft_interleaved_ranking
#print 'P credits: ', credits[0]
#print 'E credits: ', credits[1]
team_winning_algo = credits.index(max(credits))

### Probabilistic interleaved ranking ###
#prob_interleaved_ranking, credits = prob_draft_interleaving(rankings, clicks)
#print 'Ranking P: ', rankings[0]
#print 'Ranking E: ', rankings[1]
#print 'Probabilistic Interleaved ranking: ', team_draft_interleaved_ranking
#print 'P credits: ', credits[0]
#print 'E credits: ', credits[1]
#prob_winning_algo = credits.index(max(credits))

##### STEP 5 #####

## RCM ###

#def get_parameter_RCM(nr_clicks, nr_docs):
#    return nr_clicks / float(nr_docs)
    
#def predict_click_probability(ranking):
#    return probabilities
    
def is_clicked(nr_clicks, nr_docs):
    P = nr_clicks / float(nr_docs)
    flip = random.random()
    return 1 if flip < P else 0
     
def get_parameter_RCM(filename):
    nr_clicks = 0
    nr_docs = 0
    with open(filename) as f:
        data = f.readlines()
    for row in data:
        if 'C' in row:
            nr_clicks += 1
        nr_docs += len(row) - 3
    nr_docs -= nr_clicks
    return nr_clicks+500000, nr_docs

sim_clicks = []   
nr_clicks, nr_docs = get_parameter_RCM("training_data.txt")
for document in rankings[team_winning_algo]:
#    probabilities = predict_click_probability(nr_clicks, nr_docs)
    sim_clicks.append(is_clicked(nr_clicks, nr_docs))
print sim_clicks
    
    
    
