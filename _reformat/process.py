import json
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

# with open('./MELD/emotion7_train.pkl', 'rb') as f2:
#     sentiment = pickle.load(f2)

def process_data(d):
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = d['situation']
    try:
        init_intensity = int(d['survey_score']['seeker']['initial_emotion_intensity'])
    except:
        init_intensity=0
    try:
        final_intensity = int(d['survey_score']['seeker']['final_emotion_intensity'])
    except:
        #print(d)
        final_intensity=init_intensity
    try:
        relevance = int(d['survey_score']['seeker']['relevance'])
    except:
        relevance = -1
    try:
        empathy = int(d['survey_score']['seeker']['empathy'])
    except:
        empathy = -1

    d = d['dialog']
    dial = []
    for uttr in d:
        text = _norm(uttr['content'])
        role = uttr['speaker']
        if role == 'seeker':
            dial.append({
                'text': text,
                'speaker': 'usr',
                'rating': uttr['annotation'].get('feedback')
            })
        else:
            dial.append({
                'text': text,
                'speaker': 'sys',
                'strategy': uttr['annotation']['strategy'],
            })
    res = {
        'emotion_type': emotion,
        'problem_type': problem,
        'situation': situation,
        'init_intensity': init_intensity,
        'final_intensity': final_intensity,
        'relevance': relevance,
        'empathy':empathy,
        'dialog': dial,
    }
    return res

data = []

#with mp.Pool(processes=mp.cpu_count()) as pool:
with mp.Pool(processes=4) as pool:
    for e in pool.imap(process_data, tqdm.tqdm(original, total=len(original))):
        data.append(e)

emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
stress_level_change = Counter([e['init_intensity'] - e['final_intensity'] for e in data])
relevance = Counter([e['relevance'] for e in data])
empathy = Counter([e['empathy'] for e in data])
print('emotion', emotions)
print('problem', problems)
print('stress_level_change: ', stress_level_change)
print('relevance: ', relevance)
print('empathy: ', empathy)



random.shuffle(data)
dev_size = int(0.15 * len(data))
test_size = int(0.15 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
with open('./train_rl.txt', 'w') as f:
    for e in train:
        f.write(json.dumps(e) + '\n')
with open('./sample_rl.json', 'w') as f:
    json.dump(train[:10], f, ensure_ascii=False, indent=2)

print('valid', len(valid))
with open('./valid_rl.txt', 'w') as f:
    for e in valid:
        f.write(json.dumps(e) + '\n')

print('test', len(test))
with open('./test_rl.txt', 'w') as f:
    for e in test:
        f.write(json.dumps(e) + '\n')
