import textattack

class EvalGoalFunction(textattack.goal_functions.TextToTextGoalFunction):
    def __init__(self, *args_, wrapper=None, args=None, **kwargs):
        self.wrapper = wrapper
        self.args = args
        super().__init__(*args_, **kwargs)

    def _is_goal_complete(self, model_output, _):

        if self.args.goal_direction == 'up':
            return model_output - self.wrapper.original_score > self.args.goal_abs_delta
        elif self.args.goal_direction == 'down':
            return self.wrapper.original_score - model_output > self.args.goal_abs_delta
        else: 
            raise NotImplementedError

    def _get_score(self, model_output, _):
        # Maximise

        if self.args.goal_direction == 'up':
            return model_output - self.wrapper.original_score
        elif self.args.goal_direction == 'down':
            return self.wrapper.original_score - model_output
        else: 
            raise NotImplementedError