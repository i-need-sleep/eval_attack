import datetime

import torch
import numpy as np
import textattack
import evaluate
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

PRETRAINED_DIR = '../pretrained'

class BLEURTConstraint(textattack.constraints.Constraint):
    def __init__(self, checkpoint='bleurt-20-d12', mean=0, std=1, threshold=0.1):
        checkpoint = f'lucadiliello/{checkpoint}'
        
        self.bleurt = BleurtForSequenceClassification.from_pretrained(checkpoint) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bleurt.to(self.device)
        self.bleurt.eval()
        self.tokenizer = BleurtTokenizer.from_pretrained(checkpoint)

        self.model = None

        self.ref = None
        self.original_score = None
        self.compare_against_original = True
        self.threshold = threshold

        self.mean = mean
        self.std = std
    
    # Update the ref for every sample
    def set_ref(self, mt, ref):
        self.ref = ref
        self.original_score = self([mt])[0]
        return self.original_score
    
    def update_normalization(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, text_inputs):
        out = [] # [score, ...]

        with torch.no_grad():
            inputs = self.tokenizer(text_inputs, [self.ref for _ in text_inputs], padding='longest', return_tensors='pt').to(self.device)
            out = self.bleurt(**inputs).logits.flatten().cpu()
            out = (out - self.mean) / self.std
            out = out.tolist()

        return out

    def _check_constraint(self, transformed_text, current_text):
        score = self(transformed_text.text)[0]
        if abs(score - self.original_score) < self.threshold:
            return True
        return False
    
    def _check_constraint_many(self, transformed_texts, reference_text):
        scores = self([t.text for t in transformed_texts])
        out = []
        for idx, text in enumerate(transformed_texts):
            if abs(scores[idx] - self.original_score) < self.threshold:
                out.append(text) 
        return out
    
    def get_score(self, transformed_text):
        return self([transformed_text])[0]
    
class BERTScoreConstraint(textattack.constraints.Constraint):
    def __init__(self, mean=0, std=1, threshold=0.1):
        self.bertscore = evaluate.load('bertscore', experiment_id=datetime.datetime.now())

        self.model = None
        self.ref = None
        self.original_score = None
        self.compare_against_original = True
        self.threshold = threshold

        self.mean = mean
        self.std = std
    
    # Update the ref for every sample
    def set_ref(self, mt, ref):
        self.ref = ref
        self.original_score = self([mt])[0]
        return self.original_score
    
    def update_normalization(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, text_inputs):
        out = [] # [score, ...]
        scores = self.bertscore.compute(predictions = text_inputs, references = [self.ref for _ in text_inputs], lang='en')['f1']

        for score in scores:
            score = (score - self.mean) / self.std
            out.append(score)
            
        return out

    def _check_constraint(self, transformed_text, current_text):
        score = self([transformed_text.text])[0]
        if abs(score - self.original_score) < self.threshold:
            return True
        return False
    
    def _check_constraint_many(self, transformed_texts, reference_text):
        scores = self([t.text for t in transformed_texts])
        out = []
        for idx, text in enumerate(transformed_texts):
            if abs(scores[idx] - self.original_score) < self.threshold:
                out.append(text) 
        return out
    
    def get_score(self, transformed_text):
        return self([transformed_text])[0]
    
class GPTConstraint(textattack.constraints.Constraint):
    def __init__(self, threshold):
        # Use a modified version of the Huggingface implementation as the original one reloads the model for eval _compute call.
        self.perplexity = evaluate.load("./utils/perplexity.py",  module_type= "measurement", experiment_id=datetime.datetime.now())
        self.threshold = threshold

        self.mt = None
        self.original_score = None
        self.compare_against_original = True
    
    # Update the ref for each sample
    def set_ref(self, mt, _):
        self.mt = mt
        self.original_score = self.perplexity.compute(data=[mt], model_id='gpt2')['perplexities'][0]
        return self.original_score

    def _check_constraint(self, transformed_text, current_text):
        score = self.perplexity.compute(data=[transformed_text.text], model_id='gpt2')['perplexities'][0]
        if score - self.original_score < self.threshold:
            return True
        return False
    
    def _check_constraint_many(self, transformed_texts, reference_text, raw_text = False):
        if raw_text:
            scores = self.perplexity.compute(data=[t for t in transformed_texts], model_id='gpt2')['perplexities']
        else:
            scores = self.perplexity.compute(data=[t.text for t in transformed_texts], model_id='gpt2')['perplexities']
        out = []
        for idx, text in enumerate(transformed_texts):
            if scores[idx] - self.original_score < self.threshold:
                out.append(text) 
        return out
    
    def get_perplexity(self, transformed_text):
        return self.perplexity.compute(data=[transformed_text], model_id='gpt2')['perplexities'][0]
    
class EmptyConstraint(textattack.constraints.Constraint):
    def __init__(self):
        self.compare_against_original = True

    def _check_constraint(self, transformed_text, current_text):
        return True
    
    def _check_constraint_many(self, transformed_texts, reference_text):
        return transformed_texts
    
class SBERTConstraint(textattack.constraints.Constraint):
    def __init__(self, threshold):
        # Use a modified version of the Huggingface implementation as the original one reloads the model for eval _compute call.
        self.sbert = SentenceTransformer(('sentence-transformers/all-distilroberta-v1'), cache_folder=PRETRAINED_DIR)
        self.threshold = threshold

        self.mt = None
        self.compare_against_original = True
    
    # Update the ref for each sample
    def set_ref(self, mt, _):
        self.mt = mt

    def get_cos_dist(self, s1, s2):
        embs = self.sbert.encode([s1, s2])
        out = np.sum(embs[0] * embs[1])
        return out

    def _check_constraint(self, transformed_text, current_text):
        score = self.get_cos_dist(self.mt, transformed_text.text)
        if score > self.threshold:
            return True
        return False
    
    def _check_constraint_many(self, transformed_texts, reference_text, raw_text = False):
        if raw_text:
            advs = [t for t in transformed_texts]
        else:
            advs = [t.text for t in transformed_texts]
        embs = self.sbert.encode([self.mt] + advs)

        out = []
        for idx, text in enumerate(transformed_texts):
            score = np.sum(embs[0] * embs[idx + 1])
            if score > self.threshold:
                out.append(text) 
        return out
    
if __name__ == '__main__':
    constraint = SBERTConstraint(0.8)

    mt = 'Airline Emirates ordered two standard 50 Boeing 777 planes with an option of a further 20 aircraft .'
    ref = 'Emirates airline orders 50 twin-aisle Boeing 777 jetliners with an option for 20 more.'
    adv = 'Airline Canada ordered two standard 50 Boeing 777 planes with an option of a second aircraft . '

    print(1)
    constraint.set_ref(mt, ref)
    print(2)
    out = constraint._check_constraint(ref, mt)
    print(out)
    out = constraint._check_constraint(adv, mt)
    print(out)