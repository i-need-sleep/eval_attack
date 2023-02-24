from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

pretrained_dir = '../../pretrained'

download_name = 'efederici/sentence-bert-base'
save_name = 'bert-mini'


model = SentenceTransformer(download_name, cache_folder=pretrained_dir)
