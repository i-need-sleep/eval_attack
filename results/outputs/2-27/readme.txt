Data:
WMT 2017 DA zh-en

Attack method:
A genetic algorithm for word substitution as described in https://arxiv.org/pdf/1909.00986.pdf (which slightly modifies https://arxiv.org/abs/1804.07998)

poc_2017-da_bleurt_down_0.2_0.2_100_gpt2.csv
poc_2017-da_bleurt_down_0.2_0.4_100_gpt2.csv
Goal: decrease bleurt score by 0.2 / 0.4 
Constraints: sentence perplexity (GPT-2) decrease < 10, ratio of words perturbed <= 0.2, word embedding distance between original/replaced words.

poc_2017-da_gpt2_up_50.0_5.0_500_bleurt.csv
poc_2017-da_sbert_down_0.2_5.0_500_bleurt.csv
Goal: increase sentence perplexity (GPT-2) by 50 / decrease cosine distance (SBERT) by 0.4
Constraints: bleurt score absolute change < 0.1, word embedding distance between original/replaced words.

*_failed.csv
Failure cases for each setup

*.pdf
Printed visualizer outputs for each setup

poc_2017-da_bleurt_down_0.4_5.0_gpt2_probDensity.csv
Goal: decrease bleurt score by 0.4 
Constraints: sentence perplexity (GPT-2) decrease < 10, # words perturbed <= 3
n_advs is the number of sampled sentences satisfying the constraint (rejection sampling, stops after 2600 trials or when 50 constraint-satisfying samples are found)
n_successful is the number of constraint-satisfying sentences satisfying the goal.
