import datetime

import textattack
import evaluate


class BLEURTConstraint(textattack.constraints.Constraint):
    def __init__(self, threshold):
        self.bleurt = evaluate.load("bleurt", module_type="metric", experiment_id=datetime.datetime.now())
        self.threshold = threshold

        self.ref = None
        self.original_score = None
        self.compare_against_original = True
    
    # Update the ref for each sample
    def set_ref(self, mt, ref):
        self.ref = ref
        self.original_score = self.bleurt.compute(predictions = [mt], references = [ref])['scores'][0]
        return self.original_score

    def _check_constraint(self, transformed_text, current_text):
        score = self.bleurt.compute(predictions = [transformed_text], references = [self.ref])['scores'][0]
        if abs(score) < self.threshold:
            return True
        return False
    
    def _check_constraint_many(self, transformed_texts, reference_text):
        scores = self.bleurt.compute(predictions = [t.text for t in transformed_texts], references = [self.ref for _ in transformed_texts])['scores']
        return [abs(score) < self.threshold for score in scores]
    
    def get_bleurt_score(self, transformed_text):
        return self.bleurt.compute(predictions = [transformed_text], references = [self.ref])['scores'][0]
    
if __name__ == '__main__':
    constraint = BLEURTConstraint(0.1)

    mt = 'Hi! I am a machine-translated sentence.'
    ref = 'Hi! I am a human-translated sentence.'
    adv = 'Hi! I am a perturbed sentence.'

    print(1)
    constraint.set_ref(mt, ref)
    print(2)
    out = constraint._check_constraint(adv, mt)
    print(out)