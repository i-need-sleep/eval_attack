import pandas as pd

DATA_FOLDER = '../data'

# Loading data
def ted_to_lst(path='wmt-zhen-tedtalks.csv'):
    out = [] #[[mt, ref], ...]

    df = pd.read_csv(f'{DATA_FOLDER}/wmt-zhen-tedtalks.csv', header=0)

    for line_idx in range(len(df)):
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