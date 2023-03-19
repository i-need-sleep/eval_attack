import datetime

import torch
import numpy as np
import textattack
import evaluate
from sentence_transformers import SentenceTransformer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

PRETRAINED_DIR = '../pretrained'

class BleuWrapper(textattack.models.wrappers.ModelWrapper): 
    def __init__(self):
        self.bleu = evaluate.load('bleu', experiment_id=datetime.datetime.now())
        self.model = None

        self.ref = None
        self.original_score = None
    
    # Update the ref for every sample
    def set_ref(self, mt, ref):
        self.ref = ref
        self.original_score = self([mt])[0]

    def __call__(self, text_inputs):
        out = [] # [score, ...]

        for line_idx, mt in enumerate(text_inputs):
            
            score = self.bleu.compute(predictions = [mt], references = [[self.ref]])['bleu']
            out.append(score)
            

        return out

class MeteorWrapper(textattack.models.wrappers.ModelWrapper): 
    def __init__(self):
        self.meteor = evaluate.load('meteor', experiment_id=datetime.datetime.now())
        self.model = None

        self.ref = None
        self.original_score = None
    
    # Update the ref for every sample
    def set_ref(self, mt, ref):
        self.ref = ref
        self.original_score = self([mt])[0]

    def __call__(self, text_inputs):
        out = [] # [score, ...]

        for line_idx, mt in enumerate(text_inputs):

            score = self.meteor.compute(predictions = [mt], references = [self.ref])['meteor']
            out.append(score)

        return out

class BertScoreWrapper(textattack.models.wrappers.ModelWrapper): 
    def __init__(self):
        self.bertscore = evaluate.load('bertscore', experiment_id=datetime.datetime.now())
        self.model = None

        self.ref = None
        self.original_score = None

        self.mean = 0
        self.std = 1
    
    # Update the ref for every sample
    def set_ref(self, mt, ref):
        self.ref = ref
        self.original_score = self([mt])[0]
        return self.original_score

    def __call__(self, text_inputs):
        out = [] # [score, ...]

        for line_idx, mt in enumerate(text_inputs):
            
            score = self.bertscore.compute(predictions = [mt], references = [self.ref], lang='en')['f1'][0]
            score = (score - self.mean) / self.std
            out.append(score)
            
        return out

class BLEURTWrapper(textattack.models.wrappers.ModelWrapper): 
    def __init__(self, checkpoint):
        checkpoint = f'lucadiliello/{checkpoint}'
        
        self.bleurt = BleurtForSequenceClassification.from_pretrained(checkpoint) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bleurt.to(self.device)
        self.bleurt.eval()
        self.tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

        self.model = None

        self.ref = None
        self.original_score = None

        self.mean = 0
        self.std = 1
    
    # Update the ref for every sample
    def set_ref(self, mt, ref):
        self.ref = ref
        self.original_score = self([mt])[0]
        return self.original_score

    def __call__(self, text_inputs):
        out = [] # [score, ...]

        with torch.no_grad():
            inputs = self.tokenizer(text_inputs, [self.ref for _ in text_inputs], padding='longest', return_tensors='pt').to(self.device)
            out = self.bleurt(**inputs).logits.flatten().cpu()
            out = (out - self.mean) / self.std
            out = out.tolist()

        return out
    
class SBERTWrapper(textattack.models.wrappers.ModelWrapper):  
    def __init__(self):
        self.sbert = SentenceTransformer(('sentence-transformers/all-distilroberta-v1'), cache_folder=PRETRAINED_DIR)
        self.model = None

        self.mt = None
        self.original_score = 1

    def set_ref(self, mt, _):
        self.mt = mt

    def __call__(self, text_inputs):
        out = [] # [score, ...]

        for line_idx, adv in enumerate(text_inputs):
            
            score = self.get_cos_dist(self.mt, adv)
            out.append(score)
            
        return out

    def get_cos_dist(self, s1, s2):
        embs = self.sbert.encode([s1, s2])
        out = np.sum(embs[0] * embs[1])
        return out
    
class GPT2Wrapper(textattack.models.wrappers.ModelWrapper):  
    def __init__(self):
        # Use a modified version of the Huggingface implementation as the original one reloads the model for eval _compute call.
        self.perplexity = evaluate.load("./utils/perplexity.py",  module_type= "measurement", experiment_id=datetime.datetime.now())
        self.model = None

        self.mt = None
        self.original_score = 1

    def set_ref(self, mt, _):
        self.mt = mt
        self.original_score = self.perplexity.compute(data=[mt], model_id='gpt2')['perplexities'][0]

    def __call__(self, text_inputs):
        return self.perplexity.compute(data=text_inputs, model_id='gpt2')['perplexities']

if __name__ == '__main__':
    perplexity = evaluate.load("perplexity",  module_type= "measurement")
    results = perplexity.compute(data=["lorem ipsum", "Happy Birthday!", "Bienvenuem", 'Ziyu is dumb'], model_id='gpt2')
    print(results)