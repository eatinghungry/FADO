import json
from shutil import which
import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
import pickle
from collections import Counter
random.seed(13)

def _norm(x):
    return ' '.join(x.strip().split())


strategies = json.load(open('./strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}
original = json.load(open('./ESConv.json'))

intensity_changes = []
relevance_scores = []
empathy_scores = []
avg_usr_ratings = []
emo_type_list, prob_type_list = [], []
emo_type_set = set()
prob_type_set = set()

def process_data(d):
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = d['situation']

    emo_type_list.append(emotion)
    prob_type_list.append(problem)
    emo_type_set.add(emotion)
    prob_type_set.add(problem)


    try:
        init_intensity = int(d['survey_score']['seeker']['initial_emotion_intensity'])
    except:
        init_intensity=0
    try:
        final_intensity = int(d['survey_score']['seeker']['final_emotion_intensity'])
    except:
        #print(d)
        final_intensity=init_intensity+1
    try:
        relevance = int(d['survey_score']['seeker']['relevance'])
    except:
        relevance = -1
    try:
        empathy = int(d['survey_score']['seeker']['empathy'])
    except:
        empathy = -1

    intensity_changes.append(final_intensity - init_intensity) # lower is better
    relevance_scores.append(relevance)
    empathy_scores.append(empathy)

    usr_ratings = []

    d = d['dialog']
    dial = []
    for uttr in d:
        text = _norm(uttr['content'])
        role = uttr['speaker']
        if role == 'seeker':
            if uttr['annotation'].get('feedback'):
                usr_ratings.append(uttr['annotation'].get('feedback'))
    if usr_ratings:     
        usr_ratings = np.array(usr_ratings).astype(float)           
        #print(usr_ratings)
        avg_usr_ratings.append(np.mean(usr_ratings))

#with mp.Pool(processes=mp.cpu_count()) as pool:
# with mp.Pool(processes=4) as pool:
#     for e in pool.imap(process_data, tqdm.tqdm(original, total=len(original))):
#         data.append(e)

for ori in tqdm.tqdm(original, total=len(original)):
    process_data(ori)

df_dict = {'intensity_changes': intensity_changes, 
           'relevance_scores': relevance_scores,
           'empathy_scores': empathy_scores,
           'avg_usr_ratings': avg_usr_ratings,
           }

from collections import Counter
emo_counts = Counter(emo_type_list)
prob_counts = Counter(prob_type_list)
print(emo_counts)
print(prob_counts)

emo_type_dict, prob_type_dict = dict(), dict()

for n, emo in enumerate(emo_type_set):
    emo_type_dict[emo] = n

for n, prob in enumerate(prob_type_set):
    prob_type_dict[prob] = n

json_dict = json.dumps(emo_type_dict)
print(json_dict)
with open('emo_type.json', 'w') as f:
    json.dump(emo_type_dict, f, indent=2, sort_keys=True, ensure_ascii=False)

json_dict2 = json.dumps(prob_type_dict)
print(json_dict2)
with open('prob_type.json', 'w') as f:
    json.dump(prob_type_dict, f, indent=2, sort_keys=True, ensure_ascii=False)

#print(df_dict)
import pandas as pd
import pandas_profiling

df = pd.DataFrame.from_dict(df_dict)

print(df.describe())

# profile = df.profile_report(title="ESconv Dataset")
# profile.to_file(output_file="./profile2.html")

# print(intensity_changes)
# print(relevance_scores)
# print(empathy_scores)
# print(avg_usr_ratings)