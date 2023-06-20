import os
import random
import copy

import torch
import pandas as pd
import evaluate
import nltk
import numpy as np

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
        # Remove rows with dulicate idxs
        attacked_df = attacked_df.drop_duplicates(subset='idx')


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
            continue
            # metadata_file_name = 'aggregated_cs-en.csv'
            # if 'commet' in file_name:
            #     raise NotImplementedError
            # elif 'deep_word_bug' in file_name or 'input_reduction' in file_name:
            #     metadata_file_name = 'aggregated_cs-en_comet_sorted.csv'
        if 'de-en' in file_name:
            metadata_file_name = 'aggregated_de-en.csv'
            if 'sorted' in file_name:
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
    metrics = ['bleurt', 'bertscore', 'comet']
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
    df = df.reset_index()
    print(df)
    df.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_aggregated.csv')
    return df

def slice_df():
    df = pd.read_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_aggregated.csv')
    df = df.iloc[: 30000]
    
    for year in [2012, 2017, 2022]:
        slice = df[df['year'] == year]
        print(year, len(slice))
    
    for metric in ['bertscore', 'bleurt', 'comet']:
        slice = df[df['metric'] == metric]
        print(metric, len(slice))

    for method in ['input_reduction', 'clare', 'faster_genetic', 'deep_word_bug']:
        slice = df[df['attack_algo'] == method]
        print(method, len(slice))

    df.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_aggregated_sliced.csv', index=False)

class SliceMaker():
    def __init__(self, path, n_annotator=4, n_sents_per_chunk=70, n_duplicates_per_chunk=15, n_degraded_per_chunk=15):
        self.tokenizer = nltk.TreebankWordTokenizer()
        self.detoknizer = nltk.TreebankWordDetokenizer()

        self.path = path
        self.n_annotations = n_annotator
        self.n_sents_per_chunk = n_sents_per_chunk
        self.n_duplicates_per_chunk = n_duplicates_per_chunk
        self.n_degraded_per_chunk = n_degraded_per_chunk
        
        self.n_del = 4

        self.make_slices()

    def make_slices(self):
        df = pd.read_csv(self.path)

        for i in range(0, len(df), self.n_sents_per_chunk):
            chunk_idx = (str(i // self.n_sents_per_chunk)).zfill(5)
            chunk_df = df.iloc[i: i + self.n_sents_per_chunk]

            # Add duplicates and degraded sentences
            chunk_df = self.add_control_items(chunk_df)

            # Make chunks for the three-sentence and two-sentence setups
            for annotator_idx in range(1, 1 + self.n_annotations):
                self.make_three_sentence_chunk(chunk_df, chunk_idx, annotator_idx)
                # self.make_two_sentence_chunk(chunk_df, chunk_idx, annotator_idx + 6)

    def add_control_items(self, chunk):
        n_duplicates = min(self.n_duplicates_per_chunk, len(chunk))
        n_degraded = min(self.n_degraded_per_chunk, len(chunk))

        # Select indices for duplicates and degraded
        duplicate_indices = random.sample(range(len(chunk)), n_duplicates)
        degraded_indices = random.sample(range(len(chunk)), n_degraded)

        duplicates = chunk.iloc[duplicate_indices]
        degraded = chunk.iloc[degraded_indices]

        # Degrade
        for i in range(len(degraded)):
            degraded.iloc[i, degraded.columns.to_list().index('adv')] = self.degrade(degraded.iloc[i]['adv'])
            degraded.iloc[i, degraded.columns.to_list().index('mt')] = self.degrade(degraded.iloc[i]['mt'])

        # Mark the control items
        chunk.loc[:, 'control'] = ['None' for _ in range(len(chunk))]
        duplicates.loc[:, 'control'] = ['duplicate' for _ in range(len(duplicates))]
        degraded.loc[:, 'control'] = ['degrade' for _ in range(len(degraded))]

        chunk.loc[:, 'control_idx'] = ['None' for _ in range(len(chunk))]
        duplicates.loc[:, 'control_idx'] = duplicate_indices

        degraded.loc[:, 'control_idx'] = degraded_indices

        # Add to the chunk
        chunk = pd.concat([chunk, duplicates, degraded])

        # The order of the two sentence
        chunk['display_order'] = [random.randint(0, 1) for _ in range(len(chunk))]

        return chunk

    def degrade(self, sent):
        tokenized = self.tokenizer.tokenize(sent)
        n_del = min(len(tokenized) - 1, self.n_del)
        del_inds = np.random.choice([i for i in range(len(tokenized))], size=n_del, replace=False)
        tokens_out = []
        for token_idx, token in enumerate(tokenized):
            if token_idx not in del_inds:
                tokens_out.append(token)
        out = self.detoknizer.detokenize(tokens_out)
        return out
    
    def make_three_sentence_chunk(self, chunk, chunk_idx, annotator_name):
        chunk = copy.deepcopy(chunk)
        chunk['annotator_split'] = [annotator_name for _ in range(len(chunk))]

        # Shuffle the chunk
        chunk = chunk.sample(frac=1)

        # Save
        chunk.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/main_chunks/pilot_hit{chunk_idx}_threeway_annotator{str(annotator_name).zfill(2)}_semantic.csv', index=False)
        chunk.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/main_chunks/pilot_hit{chunk_idx}_threeway_annotator{str(annotator_name + 4).zfill(2)}_fluency.csv', index=False)

    def make_two_sentence_chunk(self, chunk, chunk_idx, annotator_name):
        chunk_a = copy.deepcopy(chunk)
        chunk_a['annotator_split'] = [annotator_name for _ in range(len(chunk_a))]
        chunk_a = chunk_a[chunk_a['control'] == 'None']

        chunk_b = copy.deepcopy(chunk_a)
        
        chunk_a['to_eval'] = ['mt' for _ in range(len(chunk_a))]
        chunk_b['to_eval'] = ['adv' for _ in range(len(chunk_b))]

        chunk = pd.concat([chunk_a, chunk_b])
        chunk = chunk.sample(frac=1)

        chunk_a, chunk_b = chunk.iloc[len(chunk) // 2 :] , chunk.iloc[: len(chunk) // 2]

        chunk_a = self.add_control_items(chunk_a)
        chunk_b = self.add_control_items(chunk_b)

        chunk_a = chunk_a.sample(frac=1)
        chunk_b = chunk_b.sample(frac=1)

        # Save
        chunk_a.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_chunks/pilot_hit{chunk_idx}_twowayA_annotator{str(annotator_name).zfill(2)}_semantic.csv', index=False)
        chunk_b.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_chunks/pilot_hit{chunk_idx}_twowayB_annotator{str(annotator_name).zfill(2)}_semantic.csv', index=False)
        chunk_a.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_chunks/pilot_hit{chunk_idx}_twowayA_annotator{str(annotator_name + 3).zfill(2)}_fluency.csv', index=False)
        chunk_b.to_csv(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_chunks/pilot_hit{chunk_idx}_twowayB_annotator{str(annotator_name + 3).zfill(2)}_fluency.csv', index=False)
    

if __name__ == '__main__':
    retrieve_metadata_for_annotation()
    # make_aggregated_csv_for_annotation()
    # slice_df()
    # SliceMaker(f'{uglobals.ANNOTATION_AGGREGATED_DIR}/pilot_aggregated_sliced.csv')