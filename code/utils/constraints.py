import textattack


class PerplexityConstraint(textattack.constraintsConstraint):
    def _check_constraint(self, transformed_text, current_text):
        # True: constraints are met
        print(transformed_text, current_text)
        return True