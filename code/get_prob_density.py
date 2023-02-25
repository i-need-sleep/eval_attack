import argparse

import numpy as np
import pandas
import textattack
import wordfreq 

import utils.metrics
import utils.data_utils
import utils.attack_utils
import utils.constraints

OUTPUT_DIR = '../results/outputs'

class ProbDensity():

    def __init__(self, args):
        print(args)
        self.args = args

        # Make a list of the 10k most frequent English words
        self.freq_words = wordfreq.top_n_list('en', 10000)

        # Wrap a list from zhen-tedtalks [[mt, ref], ...]
        self.pairs = utils.data_utils.ted_to_lst(args.dataset)

        # Remember to change me
        self.constraint = utils.constraints.GPTConstraint(threshold=args.gpt_constraint_threshold)

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
        elif args.victim == 'gpt2':
            wrapper = utils.metrics.GPT2Wrapper()
        else:
            raise NotImplementedError
        
        # Set up a modified goal function 
        self.goal_fn = utils.attack_utils.EvalGoalFunction(wrapper, wrapper=wrapper, args=args)
       

    def get_prob_density(self):
        out = {
            'mt': [],
            'ref': [],
            'n_advs': [],
            'n_successful': []
        }

        for pair_idx, pair in enumerate(self.pairs):
            # Get constrained sentences by rejection sampling
            advs = self.get_advs(pair)
            if len(advs) == 0:
                out.append[0, 0]

            # Check whether the advs are successful attacks
            n_success = 0
            for adv in advs:
                if self.goal_fn._is_goal_complete(adv, 0):
                    n_success += 1

            out['mt'].append(pair[0])
            out['ref'].append(pair[1])
            out['n_advs'].append(len(advs))
            out['n_successful'].append(n_success)

            # Write the output
            df = pandas.DataFrame(data=out)
            save_name = f'{args.name}_{args.dataset}_{args.victim}_{args.goal_direction}_{args.goal_abs_delta}_{args.log_prob_diff}_{args.n_samples}_{args.lm_constraint}{"_precFlipOnly" if args.only_flip_ratio_constraints else ""}_probDensity'
            print(f'Saving at {save_name}')
            df.to_csv(f'{OUTPUT_DIR}/{save_name}.csv')
        return

    def get_advs(self, pair):
        
        mt, ref = pair

        self.constraint.set_ref(mt, ref)
        filtered_sentences = []

        for _ in range(args.max_rejection_sampling_steps):
            if len(filtered_sentences) >= self.args.n_trials:
                break

            # Sample sentences given the max #perturbed words
            batch = [self.sample_sentence(mt) for __ in range(self.args.batch_size)]

            # Rejection sampling
            filtered_batch = self.constraint._check_constraint_many(batch, _, raw_text=True)
            filtered_sentences += filtered_batch

        return filtered_sentences


    def sample_sentence(self, mt):

        # Prepare the mt string
        mt_proc = textattack.shared.AttackedText(mt)

        # Select perturbed indices
        max_n_perturb = min([len(mt_proc.words), self.args.max_n_perturbed_words])
        n_perturb = np.random.choice([i + 1 for i in range(max_n_perturb)])
        perturbed_indices = np.random.choice([i for i in range(len(mt_proc.words))], size = n_perturb, replace = False).tolist()

        # Select new words from the 5k most frequent words in English
        new_words = np.random.choice(self.freq_words, size = n_perturb).tolist()

        # Swap the words
        out = mt_proc.replace_words_at_indices(perturbed_indices, new_words)

        return out.text
    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed', type=str) 

    parser.add_argument('--dataset', default='wmt-zhen-tedtalks', type=str)
    
    # Victim
    parser.add_argument('--victim', default='bleu4', type=str) 

    # Attack
    # To be implemented
    parser.add_argument('--goal_direction', default='down', type=str) 
    parser.add_argument('--goal_abs_delta', default='0.05', type=float) 

    parser.add_argument('--only_flip_ratio_constraints', action='store_true')
    parser.add_argument('--flip_max_percent', default='0.1', type=float) 
    parser.add_argument('--log_prob_diff', default='5', type=float) 
    parser.add_argument('--lm_constraint', default='google', type=str) 
    parser.add_argument('--bleurt_threshold', default=0.1, type=float) 
    parser.add_argument('--gpt_constraint_threshold', default=10, type=float) 

    # Monte Carlo
    parser.add_argument('--n_trials', default=50, type=int) 
    parser.add_argument('--max_n_perturbed_words', default=3, type=int) 
    parser.add_argument('--batch_size', default=32, type=int) # For the GPT2 perplexity constraint
    parser.add_argument('--max_rejection_sampling_steps', default=100, type=int)

    args = parser.parse_args()

    prob_dense = ProbDensity(args)
    prob_dense.get_prob_density()
