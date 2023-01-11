import argparse

import torch
import torchtext

import textattack

def make_adv(args):
    print(args)

    # Wrap a dataset from zhen-tedtalks



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Victim
    parser.add_argument('--victim', default='bleu4', type=str) 

    args = parser.parse_args()

    make_adv(args)