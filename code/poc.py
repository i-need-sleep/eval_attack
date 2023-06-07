import argparse
import os
import copy

import pandas
import torch
import textattack

import utils.metrics
import utils.data_utils
import utils.attack_utils
import utils.constraints
import utils.globals as uglobals
import attacks

def make_adv(args):
    print(args)

    # Wrap the victim metric
    if args.victim == 'bleu4':
        wrapper = utils.metrics.BleuWrapper()
    elif args.victim == 'meteor':
        wrapper = utils.metrics.MeteorWrapper()
    elif args.victim == 'bertscore':
        wrapper = utils.metrics.BertScoreWrapper()
    elif args.victim == 'bleurt':
        wrapper = utils.metrics.BLEURTWrapper(args.bleurt_checkpoint)
    elif args.victim == 'sbert':
        wrapper = utils.metrics.SBERTWrapper()
    elif args.victim == 'gpt2':
        wrapper = utils.metrics.GPT2Wrapper()
    elif args.victim == 'comet':
        torch.set_float32_matmul_precision('medium')
        wrapper = utils.metrics.COMETWrapper()
    else:
        raise NotImplementedError

    # Wrap a list from the dataset [[mt, ref], ...]
    if args.use_normalized:
        pairs, mean, std = utils.data_utils.normalized_to_list(args.dataset)
        wrapper.update_normalization(mean, std)
        print(f'Mean: {mean}, std: {std}')
    else:
        pairs = utils.data_utils.ted_to_lst(args.dataset)

    # Set up a modified goal function 
    goal_fn = utils.attack_utils.EvalGoalFunction(wrapper, wrapper=wrapper, args=args)

    # Set up constraints
    constraints = []
    constraint_update_inds = [] # Indices in the constraints list for constraints needing an update for each sentence
    constraint_write_inds = {} # {name: indice, ...}

    gpt_constraint_used = args.gpt_constraint_threshold > 0
    sbert_constraint_used = args.sbert_constraint_threshold > 0
    bleurt_constraint_used = args.bleurt_constraint_threshold > 0
    symmetric_bleurt_constraint_used = args.symmetric_bleurt_constraint_threshold > 0
    bertscore_constraint_used = args.bertscore_constraint_threshold > 0
    strict_attack = False
    if gpt_constraint_used:
        constraints.append(utils.constraints.GPTConstraint(args.gpt_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
        constraint_write_inds['gpt2_perplexity'] = len(constraints) - 1
    if sbert_constraint_used:
        constraints.append(utils.constraints.SBERTConstraint(args.sbert_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
        constraint_write_inds['sbert_cos_distance'] = len(constraints) - 1
    if bleurt_constraint_used:
        strict_attack = True
        constraints.append(utils.constraints.BLEURTConstraint(mean=mean, std=std, threshold=args.bleurt_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
        constraint_write_inds['bleurt_constraint'] = len(constraints) - 1
    if symmetric_bleurt_constraint_used:
        strict_attack = True
        constraints.append(utils.constraints.SymmetricBLEURTConstraint(mean=mean, std=std, threshold=args.bleurt_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
        constraint_write_inds['symmetric_bleurt_constraint'] = len(constraints) - 1
    if bertscore_constraint_used:
        # strict_attack = True
        constraints.append(utils.constraints.BERTScoreConstraint(mean=mean, std=std, threshold=args.bertscore_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
        constraint_write_inds['bertscore_constraint'] = len(constraints) - 1

    print(f'strict: {strict_attack}')
    # Set up the attack
    if args.attack_algo == 'faster_genetic':
        attack = attacks.FasterGeneticAlgorithm.build(constraints, goal_fn, args, strict=strict_attack)
    elif args.attack_algo == 'clare':
        attack = attacks.CLARE.build(constraints, goal_fn, args, strict=strict_attack)
    elif args.attack_algo == 'input_reduction':
        attack = attacks.InputReduction.build(constraints, goal_fn, args)
    elif args.attack_algo == 'deep_word_bug':
        attack = attacks.DeepWordBug.build(constraints, goal_fn)
    else:
        raise NotImplementedError
    
    attack.cuda_()
    # TODO: Support multiple samples. See attacker.attack_dataset()
    print(attack)

    # Set up the output file
    out = {
        'idx': [],
        'mt': [],
        'ref': [],
        'adv': [],
        'original_score': [],
        'adv_score': [],
        'score_diff': []
    }
    for key in constraint_write_inds.keys():
        out[key] = []
        out[key + '_diff'] = []
    
    # Continue from previous outputs
    covered_len = 0
    save_name = f'''\
        {args.name}\
        _{args.attack_algo}\
        _{args.dataset}\
        _{args.victim}\
        {"_" + args.bleurt_checkpoint if args.victim=="bleurt" else ""}\
        _{args.goal_direction}\
        _{args.goal_abs_delta}\
        {"_gpt"+str(args.gpt_constraint_threshold) if gpt_constraint_used else ""}\
        {"_bleurt"+str(args.bleurt_constraint_threshold) if bleurt_constraint_used else ""}\
        {"_symmetric_bleurt"+str(args.symmetric_bleurt_constraint_threshold) if symmetric_bleurt_constraint_used else ""}\
        {"_sbert"+str(args.sbert_constraint_threshold) if sbert_constraint_used else ""}\
        {"_bertscore"+str(args.bertscore_constraint_threshold) if bertscore_constraint_used else ""}\
        '''
    save_name = save_name.replace(' ', '')
    save_path = f'{uglobals.OUTPUT_DIR}/{save_name}.csv'
    if os.path.exists(save_path):
        print(f'Loaded from {save_name}')
        out, covered_len = utils.data_utils.csv_to_dict(save_name)
        failed_out, covered_len = utils.data_utils.csv_to_dict(save_name.replace('.csv, _failed.csv'))
    else:
        failed_out = copy.deepcopy(out)
    
    # Attack!
    for pair_idx, pair in enumerate(pairs):
        if pair_idx % 5 == 0:
            print(pair_idx)

        # Skip the samples already covered
        if pair_idx < covered_len:
            continue
        
        mt, ref, src = pair

        # Compute the original score
        # Update the reference
        wrapper.set_ref(mt, ref, src)
        for update_idx in constraint_update_inds:
            attack.constraints[update_idx].set_ref(mt, ref)
            
        # Run the attack
        attack_results = attack.attack(mt, 1)

        # Write the output
        # Successful attacks
        if str(type(attack_results)) == "<class 'textattack.attack_results.successful_attack_result.SuccessfulAttackResult'>":
            lines = attack_results.str_lines()
            out['adv'].append(lines[2])
            out['mt'].append(mt)
            out['ref'].append(ref)
            out['idx'].append(pair_idx)
            out['original_score'].append(wrapper.original_score)
            out['adv_score'].append(float(lines[0].split('>')[1]))
            out['score_diff'] = float(lines[0].split('>')[1]) - wrapper.original_score

            # Write the values for the constraints
            for key, idx in constraint_write_inds.items():
                constraint_val = attack.constraints[idx].get_score(lines[2]) # Against mt.
                out[key].append(constraint_val)
                out[key + '_diff'].append(constraint_val - attack.constraints[idx].original_score)
        
        # Failed attackes
        elif str(type(attack_results)) == "<class 'textattack.attack_results.failed_attack_result.FailedAttackResult'>":
            adv = attack_results.perturbed_result.attacked_text.text
            adv_score = attack_results.perturbed_result.output
            
            failed_out['adv'].append(adv)
            failed_out['mt'].append(mt)
            failed_out['ref'].append(ref)
            failed_out['idx'].append(pair_idx)
            failed_out['original_score'].append(wrapper.original_score)
            failed_out['adv_score'].append(adv_score)
            failed_out['score_diff'] = adv_score - wrapper.original_score

            # Write the values for the constraints
            for key, idx in constraint_write_inds.items():
                constraint_val = attack.constraints[idx].get_score(adv) # Against mt.
                failed_out[key].append(constraint_val)
                failed_out[key + '_diff'].append(constraint_val - attack.constraints[idx].original_score)
        else:
            raise NotImplementedError

        # Write the output for every 10 samples:
        if pair_idx % 10 == 0:
            df = pandas.DataFrame(data=out)
            print(f'Saving at {save_name}')
            df.to_csv(save_path)
            df_failed = pandas.DataFrame(data=failed_out)
            df_failed.to_csv(f'{uglobals.OUTPUT_DIR}/{save_name}_failed.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed', type=str) 

    # Data
    parser.add_argument('--dataset', default='wmt-zhen-tedtalks', type=str)
    parser.add_argument('--read_path', default='', type=str) # Continue from this output file
    parser.add_argument('--use_normalized', action='store_true')

    # Victim
    parser.add_argument('--victim', default='bleu4', type=str) 
    parser.add_argument('--bleurt_checkpoint', default='bleurt-base-128', type=str) 

    # Goal
    parser.add_argument('--goal_direction', default='down', type=str) 
    parser.add_argument('--goal_abs_delta', default='0.05', type=float) 

    # Constraints
    # Implement me
    # parser.add_argument('--flip_ratio_constraints', action='store_true')
    # parser.add_argument('--flip_max_percent', default='0.1', type=float) 
    # parser.add_argument('--max_words_perturbed_constraint', action='store_true')
    # parser.add_argument('--word_emb_constraint', action='store_true')
    # parser.add_argument('--google_lm_log_prob_diff', default='5', type=float) 
    
    # REMEMBER TO UPDATE THE CONSTRAINT
    parser.add_argument('--gpt_constraint_threshold', default=0, type=float) # 10
    parser.add_argument('--sbert_constraint_threshold', default=0, type=float) # 0.7

    parser.add_argument('--bleurt_constraint_threshold', default=0, type=float) # 0.1 * std
    parser.add_argument('--symmetric_bleurt_constraint_threshold', default=0, type=float) # 0.1 * std

    parser.add_argument('--bertscore_constraint_threshold', default=0, type=float) # 0.1 * std

    # Attack algorithm
    parser.add_argument('--attack_algo', default='faster_genetic', type=str) 

    args = parser.parse_args()

    # if args.debug:
    #     args.name = '20-d12'
    #     args.dataset = 'aggregated_de-en_bleurt-20-d12'
    #     args.use_normalized = True
    #     args.victim = 'bleurt'
    #     args.bleurt_checkpoint = 'bleurt-20-d12'
    #     args.goal_direction = 'down'
    #     args.goal_abs_delta = 1
    #     args.attack_algo = 'faster_genetic'
    #     args.bleurt_constraint_threshold = 0.5

    if args.debug:
        args.name = 'comet'
        args.dataset = 'aggregated_de-en_comet_sorted'
        args.use_normalized = True
        args.victim = 'comet'
        args.goal_direction = 'down'
        args.goal_abs_delta = 1
        args.attack_algo = 'deep_word_bug'
        args.gpt_constraint_threshold = 10

    make_adv(args)
