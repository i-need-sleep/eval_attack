import pandas as pd

DATA_FOLDER = '../data'

def ted_to_lst(path='wmt-zhen-tedtalks.csv'):
    df = pd.read_csv(f'{DATA_FOLDER}/wmt-zhen-tedtalks.csv', header=0)
    
    return