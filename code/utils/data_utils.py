import pandas as pd
import numpy as np

import  utils.globals as uglobals

DATA_FOLDER = '../data'
OUTPUT_DIR = '../results/outputs'

# Loading data
def ted_to_lst(path):
    out = [] #[[mt, ref], ...]

    df = pd.read_csv(f'{DATA_FOLDER}/{path}.csv', header=0)

    for line_idx in range(len(df)):
        if path == '2017-da':
            if df['lp'][line_idx] != 'zh-en':
                continue


        mt = df['mt'][line_idx]
        ref = df['ref'][line_idx]
        out.append([mt, ref])
        
    return out 

# Batch eval for original mt/refs
def original_scores(pairs, wrapper):
    scores = wrapper(pairs[:5])
    
    out = []
    for idx, score in enumerate(scores):
        out.append((pairs[idx][0], score))
        
    return out

def csv_to_dict(path):
    df = pd.read_csv(f'{OUTPUT_DIR}/{path}')
    df_failed = pd.read_csv(f'{OUTPUT_DIR}_failed/{path}')

    covered_len = len(df) + len(df_failed) - 2
    
    out = {}
    data = df.values.tolist()
    for key in df.keys()[1:]:
        out[key] = []
        
    for datum in data:
        for i, d in enumerate(datum):
            if i > 0:
                out[df.keys()[i]].append(d)
                
    return out, covered_len

def normalized_to_list(name): 
    #[[mt, ref], ...], mean, std
    path = f'{uglobals.PROCESSED_DIR}/{name}.csv'

    df = pd.read_csv(path)

    pairs = []
    for i in range(len(df)):
        pairs.append([df['mt'][i], df['ref'][i]])
    
    scores = np.array(df['score'])
    mean = np.mean(scores)
    std = np.std(scores)

    return pairs, mean, std