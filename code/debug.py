import os
import datetime

import evaluate
import pandas as pd

from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import comet

import utils.data_utils
import utils.globals as uglobals
from annotation_data_proc import *

# for name in os.listdir(f'{uglobals.OUTPUT_DIR}/consistency_raw'):
#     if 'reversed' not in name:
#         path = f'{uglobals.OUTPUT_DIR}/consistency_raw/{name}'
#         reversed_path = f'{uglobals.OUTPUT_DIR}/consistency_raw/{name.replace("sorted", "sorted_reversed")}'
#         if os.path.exists(reversed_path):
#             df = pd.read_csv(path)
#             reversed_df = pd.read_csv(reversed_path)
#             reversed_df['idx'] = 19000 - 1 - reversed_df['idx']

#             # Filter out duplicates
#             max_idx = df.iloc[-1]['idx']
#             reversed_df = reversed_df[reversed_df['idx'] > max_idx]
#             df_out = pd.concat([df, reversed_df])
#             df_out = df_out[df_out['bertscore_constraint_diff'] > -0.3]
#             print(df_out)
#             df_out.to_csv(f'{uglobals.OUTPUT_DIR}/consistency_reversed/{name}', index=False)

# make_aggregated_csv_for_annotation()

# paths = []
# for name in os.listdir(uglobals.ANNOTATION_METADATA_DIR):
#     if 'failed' not in name:
#         paths.append(f'{uglobals.ANNOTATION_METADATA_DIR}/{name}')
# print(paths)
# df = pd.concat([pd.read_csv(path) for path in paths])
# df.to_csv(f'{uglobals.ANNOTATION_METADATA_DIR}/consistency.csv', index=False)

original = 'There are a lot of Latin American ingredients in your Tempo and melody.'
adv = 'There are a lot of good Latin American ingredients in your Tempo and melody.'

bertscore = evaluate.load('bertscore', experiment_id=datetime.datetime.now())
score = bertscore.compute(predictions = [original], references = [original], lang='en')['f1']
print(score)

score = bertscore.compute(predictions = [original], references = [adv], lang='en')['f1']
print(score)