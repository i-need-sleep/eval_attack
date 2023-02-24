import argparse

import pandas
import textattack

import models.fast_genetic_modified

import utils.metrics
import utils.data_utils
import utils.attack_utils
import utils.constraints

OUTPUT_DIR = '../results/outputs'
CHECKPOINT_DIR = '../results/checkpoints'

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
        wrapper = utils.metrics.BLEURTWrapper()
    elif args.victim == 'sbert':
        wrapper = utils.metrics.SBERTWrapper()
    else:
        raise NotImplementedError

    # Wrap a list from zhen-tedtalks [[mt, ref], ...]
    pairs = utils.data_utils.ted_to_lst(args.dataset)

    # Set up a modified goal function 
    goal_fn = utils.attack_utils.EvalGoalFunction(wrapper, wrapper=wrapper, args=args)

    # Set up the attack
    if args.only_flip_ratio_constraints:
        attack = models.fast_genetic_modified.FasterGeneticAlgorithmJia2019MaxWordsPerturbed.build(wrapper, args)
    else:
        attack = textattack.attack_recipes.faster_genetic_algorithm_jia_2019.FasterGeneticAlgorithmJia2019.build(wrapper)

        if args.lm_constraint == 'google':
            attack.constraints[2].max_log_prob_diff = args.log_prob_diff
        elif args.lm_constraint == 'gpt2':
            attack.constraints[2] = textattack.constraints.grammaticality.language_models.GPT2(max_log_prob_diff=args.log_prob_diff, compare_against_original=True)
        elif args.lm_constraint == 'bleurt':
            attack.constraints[2] = utils.constraints.BLEURTConstraint(args.bleurt_threshold)
        else:
            raise NotImplementedError

    attack = textattack.attack.Attack(goal_fn, attack.constraints, attack.transformation, attack.search_method)

    out = {
        'mt': [],
        'ref': [],
        'adv': [],
        'original_score': [],
        'adv_score': [],
        'cos_dist': []
    }
    failed_out = []
    # Attack!
    for pair_idx, pair in enumerate(pairs):
        if pair_idx % 5 == 0:
            print(pair_idx)

        if pair_idx >= args.n_samples:
            break
        
        mt, ref = pair

        # Compute the original score
        # Update the reference
        wrapper.set_ref(mt, ref)
        if args.lm_constraint == 'bleurt':
            attack.constraints[2].set_ref(mt, ref)

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

            if args.lm_constraint == 'bleurt':
                out['original_score'].append(attack.constraints[2].original_score)
                out['adv_score'].append(attack.constraints[2].get_bleurt_score(lines[2]))
                out['cos_dist'].append(lines[0].split('>')[1])
            else:
                out['original_score'].append(wrapper.original_score)
                out['adv_score'].append(lines[0].split('>')[1])
                out['cos_dist'].append(0)
        except:
            print(lines)
            failed_out.append(lines)

        # Write the output every 10 samples:
        if pair_idx % 10 == 0:
            df = pandas.DataFrame(data=out)
            save_name = f'{args.name}_{args.dataset}_{args.victim}_{args.goal_direction}_{args.goal_abs_delta}_{args.log_prob_diff}_{args.n_samples}_{args.lm_constraint}{"_precFlipOnly" if args.only_flip_ratio_constraints else ""}'
            df.to_csv(f'{OUTPUT_DIR}/{save_name}.csv')
            df_failed = pandas.DataFrame(data=failed_out)
            df_failed.to_csv(f'{OUTPUT_DIR}/{save_name}_failed.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed', type=str) 

    parser.add_argument('--dataset', default='wmt-zhen-tedtalks', type=str)
    
    # Victim
    parser.add_argument('--victim', default='bleu4', type=str) 

    # Attack
    parser.add_argument('--n_samples', default='500', type=int)

    parser.add_argument('--goal_direction', default='down', type=str) 
    parser.add_argument('--goal_abs_delta', default='0.05', type=float) 

    parser.add_argument('--only_flip_ratio_constraints', action='store_true')
    parser.add_argument('--flip_max_percent', default='0.1', type=float) 
    parser.add_argument('--log_prob_diff', default='5', type=float) 
    parser.add_argument('--lm_constraint', default='google', type=str) 
    parser.add_argument('--bleurt_threshold', default=0.1, type=float) 

    args = parser.parse_args()

    make_adv(args)
