import pandas as pd

DATA_FOLDER = '../data'

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
    print(out)
    return out 

# Batch eval for original mt/refs
def original_scores(pairs, wrapper):
    scores = wrapper(pairs[:5])
    
    out = []
    for idx, score in enumerate(scores):
        out.append((pairs[idx][0], score))
        
    return out