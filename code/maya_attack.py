import pandas as pd
import os
from models.maya.MG.rl import Agent
from models.maya.models.models import VictimBertForSequenceClassification, VictimBLEURT
from models.maya.MG.multi_granularity import MGAttacker
from models.maya.MG.paraphrase_methods import T5, GPT2Paraphraser, BackTranslation
from models.maya.MG.substitute_methods import SubstituteWithBert
from tqdm import tqdm

import argparse
import utils.data_utils
import torch


def attack():
    if os.path.exists(dest_path):
        attack_samples = pd.read_csv(dest_path, sep='\t')
        count = attack_samples['index'].values[-1] + 1
    else:
        attack_samples = pd.DataFrame(columns=['index', 'ori', 'adv', 'substitution', 'paraphrase', 'query'])
        count = 0

    total = len(mts)
    print(total)
    print(count)
    # show the information about attacking
    progress_bar = tqdm(range(count, total))
    suc = len(attack_samples.values)
    fail = count - suc
    attack_num = count
    suc_rate = suc / attack_num * 100 if attack_num != 0 else 0

    for i in progress_bar:
        progress_bar.set_description(
            '\033[0;31mparaphase:{} suc:{}  fail:{} total:{}  suc_rate:{:.2f}%\033[0m'.format(
                paraphrases, suc, fail, attack_num, suc_rate))
        
        victim_model.set_ref(mts[i], refs[i])

        info = attacker.attack(mts[i], labels[i])

        if info['adv'] is None:
            fail += 1
            attack_num += 1
            suc_rate = suc / attack_num * 100

        else:
            suc += 1
            attack_num += 1
            suc_rate = suc / attack_num * 100
            ori = mts[i][0] if isinstance(mts[i], list) else mts[i]
            attack_samples = attack_samples.append({'index': i,
                                                    'ori': ori,
                                                    'adv': info['adv'],
                                                    'substitution': info['substitution'],
                                                    'paraphrase': info['paraphrase'],
                                                    'query': info['query']}, ignore_index=True)

        if attack_samples.shape[0] > 0:
            attack_samples.to_csv(dest_path, sep='\t', index=False)
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bleurt_checkpoint', default='bleurt-base-128', type=str) 

    # Goal
    parser.add_argument('--goal_direction', default='down', type=str) 
    parser.add_argument('--goal_abs_delta', default='0.2', type=float) 

    args = parser.parse_args()
    print(args)

    # fill your own dataset path
    print("load dataset")
    pairs = utils.data_utils.ted_to_lst('2017-da')  # [[mt, ref], ...]

    mts = [pair[0] for pair in pairs]
    refs = [pair[1] for pair in pairs]
    labels = [0 for _ in pairs]

    # fill your own victim model path
    print("load victim model")
    victim_model = VictimBLEURT(args)

    # choose your paraphrase models
    paraphrases = ['t5']
    paraphrase_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'bt' in paraphrases:
        # use back translation
        print("initialize back tran")
        paraphrase_list.append(BackTranslation())

    if 't5' in paraphrases:
        # use T5
        print("initialize T5")
        paraphrase_list.append(T5(device))

    if 'gpt2' in paraphrases:
        # use gpt2 paraphrase model
        # fill your own gpt2 model path
        print("initialize gpt2")
        gpt2_path = 'paraphrase_models/style_transfer_paraphrase/paraphraser_gpt2_large'
        paraphrase_list.append(GPT2Paraphraser(gpt2_path, 'cuda:1'))

    # choose your paraphrase models
    print("initialize substitution model")
    substitution = SubstituteWithBert(victim_model, device)

    # output result
    dest_path = f'../results/outputs/MAYA_{args.bleurt_checkpoint}_{args.goal_direction}_{args.goal_abs_delta}.tsv'

    attack_times = 10

    print("initialize attacker")
    # if use rl
    # attack_model = BertForSequenceClassification.from_pretrained(
    #     '../models/pretrained_models/mayapi_bert_for_sst2').to('cuda:0')
    # attack_model.eval()
    # agent = Agent(attack_model)
    # attacker = RLMGAttacker(attack_times, victim_model, substitution, paraphrase_list, agent)
    
    attacker = MGAttacker(attack_times, victim_model, substitution, paraphrase_list)

    # start attack
    print("start attack")
    attack()
