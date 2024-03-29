import random 
import os

import pandas as pd
import numpy as np
import tqdm

import utils.globals as uglobals
import utils.metrics

def preproc_wmt_year(src_path, ref_path, mt_dir, year, sys_name_idx, subsample=True):

    # Output df cols:
    # year, src-ref langs, system, mt, ref
    year_out, src_ref, mt_sys, mt_out, ref_out, src_out = [], [], [], [], [], []

    # Subsample refs
    with open(ref_path, 'r', encoding='utf-8') as f:
        refs = f.readlines()
    with open(src_path, 'r', encoding='utf-8') as f:
        srcs = f.readlines()
    
    # Filter out short refs
    ref_indices = []
    for ref_idx, ref in enumerate(refs):
        if len(ref.split(' ')) >= MIN_REF_LEN:
            ref_indices.append(ref_idx)
    
    if not subsample:
        global N_SENTS
        N_SENTS = len(ref_indices)

    indices = random.sample(ref_indices, N_SENTS)
    
    # Fetch the mts
    systems = os.listdir(mt_dir)
    for system in systems:

        year_out += [year for _ in range(N_SENTS)]
        src_ref += [f'{SRC_LANG}-{REF_LANG}' for _ in range(N_SENTS)]
        mt_sys += [system.split('.')[sys_name_idx] for _ in range(N_SENTS)]

        path = f'{mt_dir}/{system}'
        with open(path, 'r', encoding='utf-8') as f:
            mts = f.readlines()
        for indice in indices:
            mt_out.append(mts[indice].strip('\n'))
            ref_out.append(refs[indice].strip('\n'))
            src_out.append(srcs[indice].strip('\n'))
        
    df = pd.DataFrame({
        'year': year_out,
        'src-ref langs': src_ref,
        'mt_sys': mt_sys,
        'mt': mt_out,
        'ref': ref_out,
        'src': src_out
    })

    save_path = f'{uglobals.PROCESSED_DIR}/{year}_{SRC_LANG}-{REF_LANG}.csv'
    print(f'Saving at: {save_path}')
    df.to_csv(save_path)
    return df

def preproc_wmt(subsample=True):
    dfs = []

    year = 2012
    src_path = f'{uglobals.DATA_DIR}/wmt12-data/plain/sources/newstest2012-src.{SRC_LANG}'
    ref_path = f'{uglobals.DATA_DIR}/wmt12-data/plain/references/newstest2012-ref.{REF_LANG}'
    mt_dir = f'{uglobals.DATA_DIR}/wmt12-data/plain/system-outputs/newstest2012/{SRC_LANG}-{REF_LANG}/'
    df = preproc_wmt_year(src_path, ref_path, mt_dir, year, -1, subsample=subsample)
    dfs.append(df)

    year = 2017
    src_path = f'{uglobals.DATA_DIR}/wmt17-metrics-task/wmt17-submitted-data/txt/sources/newstest2017-{SRC_LANG}{REF_LANG}-src.{SRC_LANG}'
    ref_path = f'{uglobals.DATA_DIR}/wmt17-metrics-task/wmt17-submitted-data/txt/references/newstest2017-{SRC_LANG}{REF_LANG}-ref.{REF_LANG}'
    mt_dir = f'{uglobals.DATA_DIR}/wmt17-metrics-task/wmt17-submitted-data/txt/system-outputs/newstest2017/{SRC_LANG}-{REF_LANG}/'
    df = preproc_wmt_year(src_path, ref_path, mt_dir, year, 1, subsample=subsample)
    dfs.append(df)

    year = 2022
    src_path = f'{uglobals.DATA_DIR}/wmt22-metrics-inputs-v7/wmt22-metrics-inputs-v6/metrics_inputs/txt/generaltest2022/sources/generaltest2022.{SRC_LANG}-{REF_LANG}.src.{SRC_LANG}'
    ref_path = f'{uglobals.DATA_DIR}/wmt22-metrics-inputs-v7/wmt22-metrics-inputs-v6/metrics_inputs/txt/generaltest2022/references/generaltest2022.{SRC_LANG}-{REF_LANG}.ref.refA.{REF_LANG}'
    mt_dir = f'{uglobals.DATA_DIR}/wmt22-metrics-inputs-v7/wmt22-metrics-inputs-v6/metrics_inputs/txt/generaltest2022/system_outputs/{SRC_LANG}-{REF_LANG}/'
    df = preproc_wmt_year(src_path, ref_path, mt_dir, year, -2, subsample=subsample)
    dfs.append(df)

    df_out = pd.concat(dfs)

    save_path = f'{uglobals.PROCESSED_DIR}/aggregated_{SRC_LANG}-{REF_LANG}_ref.csv'
    print(f'Saving at: {save_path}')
    df_out.to_csv(save_path, index=False)

def eval_preproced(preproced_path, metric_name, normalization):
    # Evaluate all pairs and normalize scores

    # Setup the metric
    if metric_name == 'bleurt-20-d12':
        metric = utils.metrics.BLEURTWrapper('bleurt-20-d12')
    elif metric_name == 'bertscore':
        metric = utils.metrics.BertScoreWrapper()
    elif metric_name == 'comet':
        metric = utils.metrics.COMETWrapper()
    else:
        raise NotImplementedError

    df = pd.read_csv(preproced_path)
    scores = []

    for idx in tqdm.tqdm(range(len(df))):
        mt, ref, src = df['mt'][idx], df['ref'][idx], df['src'][idx]
        score = metric.set_ref(mt, ref, src)
        scores.append(score)
    
    scores = np.array(scores)

    if normalization == 'std':
        norm_scores = (scores - np.mean(scores)) / np.std(scores)
    elif normalization == '01':
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    else:
        raise NotImplementedError

    df['score'] = scores
    df['normalized_score'] = norm_scores
    
    save_path = f'{uglobals.PROCESSED_DIR}/aggregated_{SRC_LANG}-{REF_LANG}_{metric_name}.csv'
    print(f'Saving at: {save_path}')
    df.to_csv(save_path, index=False)
    return

def fetch_src(src_name, file_name):
    df = pd.read_csv(f'{uglobals.PROCESSED_DIR}/{file_name}.csv')
    src_df = pd.read_csv(f'{uglobals.PROCESSED_DIR}/{src_name}.csv')
    srcs = []

    for ref in tqdm.tqdm(df['ref']):
        for idx, src_ref in enumerate(src_df['ref']):
            if ref == src_ref:
                srcs.append(src_df['src'][idx])
                break
    
    df['src'] = srcs
    df.to_csv(f'{uglobals.PROCESSED_DIR}/{file_name}.csv')

if __name__ == '__main__':
    
    SRC_LANG = 'de'
    REF_LANG = 'en'
    # N_SENTS = 100
    MIN_REF_LEN = 0

    preproc_wmt(False)

    # eval_preproced(f'{uglobals.PROCESSED_DIR}/aggregated_{SRC_LANG}-{REF_LANG}.csv', 'bleurt-20-d12', 'std')
    # eval_preproced(f'{uglobals.PROCESSED_DIR}/aggregated_{SRC_LANG}-{REF_LANG}.csv', 'bertscore', 'std')
    # eval_preproced(f'{uglobals.PROCESSED_DIR}/aggregated_{SRC_LANG}-{REF_LANG}.csv', 'comet', 'std')

    # fetch_src(f'aggregated_{SRC_LANG}-{REF_LANG}_ref', f'aggregated_{SRC_LANG}-{REF_LANG}')