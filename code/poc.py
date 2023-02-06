import argparse

import pandas
import textattack

import models.fast_genetic_modified

import utils.metrics
import utils.data_utils
import utils.attack_utils

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
    else:
        raise NotImplementedError

    # Wrap a list from zhen-tedtalks [[mt, ref], ...]
    pairs = utils.data_utils.ted_to_lst()

    # Set up a modified goal function 
    goal_fn = utils.attack_utils.EvalGoalFunction(wrapper, wrapper=wrapper, args=args)

    # Set up the attack
    if args.only_flip_ratio_constraints:
        attack = models.fast_genetic_modified.FasterGeneticAlgorithmJia2019MaxWordsPerturbed.build(wrapper, args)
    else:
        attack = textattack.attack_recipes.faster_genetic_algorithm_jia_2019.FasterGeneticAlgorithmJia2019.build(wrapper)

    attack = textattack.attack.Attack(goal_fn, attack.constraints, attack.transformation, attack.search_method)

    print(attack)

    out = {
        'mt': [],
        'ref': [],
        'adv': [],
        'original_score': [],
        'adv_score': [],
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

        # Run the attack
        attack_results = attack.attack(mt, 1)
        lines = attack_results.str_lines()

        # Write the output
        try:
            out['adv'].append(lines[2])
            out['mt'].append(mt)
            out['ref'].append(ref)
            out['original_score'].append(wrapper.original_score)
            out['adv_score'].append(lines[0].split('>')[1])
        except:
            print(lines)
            failed_out.append(lines)

    df = pandas.DataFrame(data=out)
    save_name = f'{args.name}_{args.victim}_{args.goal_direction}_{args.goal_abs_delta}_{args.n_samples}{"_precFlipOnly" if args.only_flip_ratio_constraints else ""}'
    df.to_csv(f'{OUTPUT_DIR}/{save_name}.csv')
    df_failed = pandas.DataFrame(data=failed_out)
    df_failed.to_csv(f'{OUTPUT_DIR}/{save_name}_failed.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed', type=str) 

    # Victim
    parser.add_argument('--victim', default='bleu4', type=str) 

    # Attack
    parser.add_argument('--n_samples', default='1000', type=int)

    parser.add_argument('--goal_direction', default='down', type=str) 
    parser.add_argument('--goal_abs_delta', default='0.05', type=float) 

    parser.add_argument('--only_flip_ratio_constraints', action='store_true')
    parser.add_argument('--flip_max_percent', default='0.1', type=float) 

    args = parser.parse_args()

    make_adv(args)
