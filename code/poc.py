import argparse

import pandas
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
    else:
        raise NotImplementedError

    # Wrap a list from the dataset [[mt, ref], ...]
    if args.use_normalized:
        pairs, mean, std = utils.data_utils.normalized_to_list(args.dataset)
        wrapper.update_normalization(mean, std)
    else:
        pairs = utils.data_utils.ted_to_lst(args.dataset)

    # Set up a modified goal function 
    goal_fn = utils.attack_utils.EvalGoalFunction(wrapper, wrapper=wrapper, args=args)

    # Set up constraints
    constraints = []
    constraint_update_inds = []
    bleurt_constraint_idx = -1
    gpt_constraint_used = args.gpt_constraint_threshold > 0
    sbert_constraint_used = args.sbert_constraint_threshold > 0
    bleurt_constraint_used = args.bleurt_constraint_threshold > 0
    bertscore_constraint_used = args.bertscore_constraint_threshold > 0
    if gpt_constraint_used:
        constraints.append(utils.constraints.GPTConstraint(args.gpt_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
    if sbert_constraint_used:
        constraints.append(utils.constraints.SBERTConstraint(args.sbert_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
    if bleurt_constraint_used:
        constraints.append(utils.constraints.BLEURTConstraint(mean=args.bleurt_constraint_mean, std=args.bleurt_constraint_std, threshold=args.bleurt_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)
        bleurt_constraint_idx = len(constraints) - 1
    if bertscore_constraint_used:
        constraints.append(utils.constraints.BERTScoreConstraint(mean=args.bertscore_constraint_mean, std=args.bertscore_constraint_std, threshold=args.bertscore_constraint_threshold))
        constraint_update_inds.append(len(constraints) - 1)

    # Set up the attack
    if args.attack_algo == 'faster_genetic':
        attack = attacks.FasterGeneticAlgorithm.build(constraints, goal_fn, args)
    elif args.attack_algo == 'clare':
        attack = attacks.CLARE.build(constraints, goal_fn, args)
    elif args.attack_algo == 'input_reduction':
        attack = attacks.InputReduction.build(constraints, goal_fn, args)
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
    }
    
    covered_len = 0
    if args.read_path != '':
        out, covered_len = utils.data_utils.csv_to_dict(args.read_path)

    if args.bleurt_constraint_threshold > 0:
        out['cos_dist'] = []
    failed_out = []
    
    # Attack!
    for pair_idx, pair in enumerate(pairs):
        if pair_idx % 5 == 0:
            print(pair_idx)

        # Skip the samples already covered
        if pair_idx < covered_len:
            continue
        
        mt, ref = pair

        # Compute the original score
        # Update the reference
        wrapper.set_ref(mt, ref)
        for update_idx in constraint_update_inds:
            attack.constraints[update_idx].set_ref(mt, ref)

        # Run the attack
        attack_results = attack.attack(mt, 1)
        lines = attack_results.str_lines()
        
        if args.debug:
            print(lines)

        # Write the output
        try:
            out['adv'].append(lines[2])
            out['mt'].append(mt)
            out['ref'].append(ref)
            out['idx'].append(pair_idx)

            # if bleurt_constraint_idx > -1: # Swapped goal/constraints
            #     out['original_score'].append(attack.constraints[bleurt_constraint_idx].original_score)
            #     out['adv_score'].append(attack.constraints[bleurt_constraint_idx].get_score(lines[2]))
            #     out['cos_dist'].append(lines[0].split('>')[1])
            # else:
            out['original_score'].append(wrapper.original_score)
            out['adv_score'].append(lines[0].split('>')[1])
        except:
            print(lines)
            failed_out.append([pair_idx, lines])

        # Write the output for every 10 samples:
        if pair_idx % 10 == 0:
            df = pandas.DataFrame(data=out)
            save_name = f'{args.name}_{args.attack_algo}_{args.dataset}_{args.victim}{"_" + args.bleurt_checkpoint if args.victim=="bleurt" else ""}_{args.goal_direction}_{args.goal_abs_delta}{"_gpt"+str(args.gpt_constraint_threshold) if gpt_constraint_used else ""}{"_bleurt" if bleurt_constraint_used else ""}{"_sbert"+str(args.sbert_constraint_threshold) if sbert_constraint_used else ""}{"_sbert"+str(args.bertscore_constraint_threshold) if bertscore_constraint_used else ""}'
            print(f'Saving at {save_name}')
            df.to_csv(f'{uglobals.OUTPUT_DIR}/{save_name}.csv')
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
    parser.add_argument('--bleurt_constraint_mean', default=0, type=float) 
    parser.add_argument('--bleurt_constraint_std', default=0, type=float) 

    parser.add_argument('--bertscore_constraint_threshold', default=0, type=float) # 0.1 * std
    parser.add_argument('--bertscore_constraint_mean', default=0, type=float) # 0.1 * std
    parser.add_argument('--bertscore_constraint_std', default=0, type=float) # 0.1 * std

    # Attack algorithm
    parser.add_argument('--attack_algo', default='faster_genetic', type=str) 

    args = parser.parse_args()

    make_adv(args)
