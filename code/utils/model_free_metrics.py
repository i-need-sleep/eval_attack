import textattack
import evaluate

class BleuWrapper(textattack.models.wrappers.ModelWrapper): 
    def __init__(self):
        self.bleu = evaluate.load('bleu')
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
        self.meteor = evaluate.load('meteor')
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