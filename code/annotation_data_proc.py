import os
import random

import torch
import pandas as pd
import evaluate

import utils.globals as uglobals

def check_perplexity_constraints(path, size=200, diff=10):
    df = pd.read_csv(path)

    # Select the indices for the sanity check
    indices = [i for i in range(len(df))]
    random.shuffle(indices)
    indices = indices[ :size]
    mt = df.iloc[indices]['mt'].tolist()
    adv = df.iloc[indices]['adv'].tolist()

    # Load the gpt2
    perplexity = evaluate.load("./utils/perplexity.py",  module_type= "measurement")
    
    mt_perplexity = perplexity.compute(data=mt, model_id='gpt2')['perplexities']
    adv_perplexity = perplexity.compute(data=adv, model_id='gpt2')['perplexities']

    mt_perplexity = torch.tensor(mt_perplexity)
    adv_perplexity = torch.tensor(adv_perplexity)

    if torch.any(adv_perplexity - mt_perplexity > diff):
        print(mt_perplexity)
        print(adv_perplexity)
        print(path)
        raise

    return

def check_perplexity_constraints_for_annotation():
    file_names = os.listdir(uglobals.ANNOTATION_RAW_DIR)
    for file_name in file_names:
        if 'failed' in file_name or 'genetic' in file_name:
            continue
        if not 'deep_word_bug' in file_name:
            continue
        print(file_name)
        check_perplexity_constraints(f'{uglobals.ANNOTATION_RAW_DIR}/{file_name}')
    return


def retrieve_metadata(attacked_path, metadata_path, out_path):
    attacked_df = pd.read_csv(attacked_path)
    metadata_df = pd.read_csv(metadata_path)

    
    if 'mt' in attacked_df.columns:
        metadata_indices = attacked_df['idx'].tolist()
        metadata_lines = metadata_df.iloc[metadata_indices]
        assert metadata_lines['mt'].tolist() == attacked_df['mt'].tolist()
    elif len(attacked_df) == 0:
        attacked_df.to_csv(out_path, index=False)
        return
    else:
        metadata_indices = attacked_df['0'].tolist()
        metadata_lines = metadata_df.iloc[metadata_indices]

        scores, mts = [], []
        for i in range(len(attacked_df)):
            score, mt = attacked_df.iloc[i]['1'].split(" --> [FAILED]', ")
            score = score[2:]
            mt = mt[1: -1]
            scores.append(score)
            mts.append(mt)
        attacked_df['mt'] = mt
        attacked_df['original_score'] = score
        


    attacked_df['year'] = metadata_lines['year'].tolist()
    attacked_df['src-ref langs'] = metadata_lines['src-ref langs'].tolist()
    attacked_df['mt_sys'] = metadata_lines['mt_sys'].tolist()
    attacked_df['src'] = metadata_lines['src'].tolist()
    attacked_df.to_csv(out_path, index=False)

def retrieve_metadata_for_annotation():
    file_names = os.listdir(uglobals.ANNOTATION_RAW_DIR)
    for file_name in file_names:
        if 'cs-en' in file_name:
            metadata_file_name = 'aggregated_cs-en.csv'
            if 'commet' in file_name:
                raise NotImplementedError
            elif 'deep_word_bug' in file_name or 'input_reduction' in file_name:
                metadata_file_name = 'aggregated_cs-en_comet_sorted.csv'
        if 'de-en' in file_name:
            metadata_file_name = 'aggregated_de-en.csv'
            if 'commet' in file_name:
                raise NotImplementedError
            elif 'deep_word_bug' in file_name or 'input_reduction' in file_name:
                metadata_file_name = 'aggregated_de-en_comet_sorted.csv'
        
        print(file_name, metadata_file_name)
        retrieve_metadata(f'{uglobals.ANNOTATION_RAW_DIR}/{file_name}', f'{uglobals.PROCESSED_DIR}/{metadata_file_name}', f'{uglobals.ANNOTATION_METADATA_DIR}/{file_name}')
    return

def sort_df(df):
    tups = [tup for tup in zip(df['year'].tolist(), df['mt_sys'].tolist(), df['metric'].tolist(), df['attack_algo'].tolist())]
    df['year-mt_sys-metric-attack_algo'] = tups
    tups = list(set(tups))
    print('#Combinations:', len(tups))

    slices = []
    for tup in tups:
        df_slice = df[df['year-mt_sys-metric-attack_algo'] == tup]
        slices.append(df_slice)

    out = slices[-1].iloc[:1]
    slices[-1] = slices[-1].drop(slices[-1].index[0])
    ctr = 0

    while True:
        all_empty = True
        for df_slice in slices:
            if len(df_slice) != 0:
                all_empty = False
        if all_empty:
            break

        for idx, df_slice in enumerate(slices):
            if len(df_slice) == 0:
                continue
            out_line = df_slice.iloc[:1]
            slices[idx] = df_slice.drop(df_slice.index[0])
            out = pd.concat([out, out_line])    

        if len(out) > ctr + 1000:
            print(len(out))  
            ctr = len(out)
            s = 0
            for df_slice in slices:
                s += len(df_slice)
            print(s)

    return out

def make_aggregated_csv_for_annotation():
    # Aggregated everything into a large csv file

    # Filter out the language pairs/metrics/attacks we care about
    langs = ['de-en']
    metrics = ['bleurt', 'bertscore']
    attack_algos = ['clare', 'deep_word_bug', 'faster_genetic', 'input_reduction']

    dfs = []
    for file_name in os.listdir(uglobals.ANNOTATION_METADATA_DIR):
        if 'failed' in file_name:
            continue
        for lang in langs:
            for metric in metrics:
                for attack_algo in attack_algos:
                    if lang in file_name and metric in file_name and attack_algo in file_name:
                        df = pd.read_csv(f'{uglobals.ANNOTATION_METADATA_DIR}/{file_name}')
                        df['metric'] = [metric for _ in range(len(df))]
                        df['attack_algo'] = [attack_algo for _ in range(len(df))]
                        dfs.append(df)
                        print(file_name)
    assert len(dfs) == len(langs) * len(metrics) * len(attack_algos)

    df = pd.concat(dfs)
    df = sort_df(df)
    print(df)
    df.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_aggregated.csv')
    return df



if __name__ == '__main__':
    make_aggregated_csv_for_annotation()