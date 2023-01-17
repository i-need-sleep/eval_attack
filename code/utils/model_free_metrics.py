from torchtext.data.metrics import bleu_score
from torchtext.data import get_tokenizer

import textattack

class BleuWrapper(textattack.models.wrappers.ModelWrapper): 
    def __init__(self, n=4):
        self.n = n
        self.tokenizer = get_tokenizer("basic_english")
        self.model = None

        self.ref_tokens = None
        self.original_score = None
    
    # Update the ref for every sample
    def set_ref(self, mt, ref):
        self.ref_tokens = self.tokenizer(ref)
        self.original_score = self([mt])[0]

    def __call__(self, text_inputs):
        out = [] # [[score], ...]

        for line_idx, mt in enumerate(text_inputs):

            mt_tokens = self.tokenizer(mt)
            
            bleu = bleu_score([mt_tokens], [[self.ref_tokens]])
            out.append(bleu)
            
        return out