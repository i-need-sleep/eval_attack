"""
Modified from textattack
Faster Alzantot Genetic Algorithm
===================================
(Certified Robustness to Adversarial Word Substitutions)


"""

from textattack import Attack
from textattack.constraints.grammaticality.language_models import (
    LearningToWriteLanguageModel,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import AlzantotGeneticAlgorithm
from textattack.transformations import WordSwapEmbedding

from textattack.attack_recipes import AttackRecipe

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordInsertionMaskedLM,
    WordMergeMaskedLM,
    WordSwapMaskedLM,
)
import transformers
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordDeletion


class FasterGeneticAlgorithm(AttackRecipe):
    @staticmethod
    def build(constraints_in, goal_fn, args):
        #
        # Section 5: Experiments
        #
        # We base our sets of allowed word substitutions S(x, i) on the
        # substitutions allowed by Alzantot et al. (2018). They demonstrated that
        # their substitutions lead to adversarial examples that are qualitatively
        # similar to the original input and retain the original label, as judged
        # by humans. Alzantot et al. (2018) define the neighbors N(w) of a word w
        # as the n = 8 nearest neighbors of w in a “counter-fitted” word vector
        # space where antonyms are far apart (Mrksiˇ c´ et al., 2016). The
        # neighbors must also lie within some Euclidean distance threshold. They
        # also use a language model constraint to avoid nonsensical perturbations:
        # they allow substituting xi with x˜i ∈ N(xi) if and only if it does not
        # decrease the log-likelihood of the text under a pre-trained language
        # model by more than some threshold.
        #
        # We make three modifications to this approach:
        #
        # First, in Alzantot et al. (2018), the adversary
        # applies substitutions one at a time, and the
        # neighborhoods and language model scores are computed.
        # Equation (4) must be applied before the model
        # can combine information from multiple words, but it can
        # be delayed until after processing each word independently.
        # Note that the model itself classifies using a different
        # set of pre-trained word vectors; the counter-fitted vectors
        # are only used to define the set of allowed substitution words.
        # relative to the current altered version of the input.
        # This results in a hard-to-define attack surface, as
        # changing one word can allow or disallow changes
        # to other words. It also requires recomputing
        # language model scores at each iteration of the genetic
        # attack, which is inefficient. Moreover, the same
        # word can be substituted multiple times, leading
        # to semantic drift. We define allowed substitutions
        # relative to the original sentence x, and disallow
        # repeated substitutions.
        #
        # Second, we use a faster language model that allows us to query
        # longer contexts; Alzantot et al. (2018) use a slower language
        # model and could only query it with short contexts.

        # Finally, we use the language model constraint only
        # at test time; the model is trained against all perturbations in N(w). This encourages the model to be
        # robust to a larger space of perturbations, instead of
        # specializing for the particular choice of language
        # model. See Appendix A.3 for further details. [This is a model-specific
        # adjustment, so does not affect the attack recipe.]
        #
        # Appendix A.3:
        #
        # In Alzantot et al. (2018), the adversary applies replacements one at a
        # time, and the neighborhoods and language model scores are computed
        # relative to the current altered version of the input. This results in a
        # hard-to-define attack surface, as the same word can be replaced many
        # times, leading to semantic drift. We instead pre-compute the allowed
        # substitutions S(x, i) at index i based on the original x. We define
        # S(x, i) as the set of x_i ∈ N(x_i) such that where probabilities are
        # assigned by a pre-trained language model, and the window radius W and
        # threshold δ are hyperparameters. We use W = 6 and δ = 5.
        #
        #
        # Swap words with their embedding nearest-neighbors.
        #
        # Embedding: Counter-fitted Paragram Embeddings.
        #
        # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
        #
        transformation = WordSwapEmbedding(max_candidates=8)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Maximum words perturbed percentage of 20%
        #
        # constraints.append(MaxWordsPerturbed(max_percent=args.flip_max_percent))
        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        # constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        #
        # Language Model
        #
        #
        #
        # constraints.append(
        #     LearningToWriteLanguageModel(
        #         window_size=6, max_log_prob_diff=5.0, compare_against_original=True
        #     )
        # )
        # constraints.append(LearningToWriteLanguageModel(window_size=5))
        #
        for constraint in constraints_in:
            constraints.append(constraint)
        # Goal is untargeted classification
        #
        # goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = AlzantotGeneticAlgorithm(
            pop_size=60, max_iters=40, post_crossover_check=False
        )

        return Attack(goal_fn, constraints, transformation, search_method)

class CLARE(AttackRecipe):
    @staticmethod
    def build(constraints_in, goal_fn, args):
        # "This paper presents CLARE, a ContextuaLized AdversaRial Example generation model
        # that produces fluent and grammatical outputs through a mask-then-infill procedure.
        # CLARE builds on a pre-trained masked language model and modifies the inputs in a context-aware manner.
        # We propose three contex-tualized  perturbations, Replace, Insert and Merge, allowing for generating outputs of
        # varied lengths."
        #
        # "We  experiment  with  a  distilled  version  of RoBERTa (RoBERTa_{distill}; Sanh et al., 2019)
        # as the masked language model for contextualized infilling."
        # Because BAE and CLARE both use similar replacement papers, we use BAE's replacement method here.

        shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(
            "distilroberta-base"
        )
        shared_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "distilroberta-base"
        )
        transformation = CompositeTransformation(
            [
                WordSwapMaskedLM(
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-4,
                ),
                WordInsertionMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=0.0,
                ),
                WordMergeMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-3,
                ),
            ]
        )

        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = [RepeatModification(), StopwordModification()]

        # "A  common  choice  of sim(·,·) is to encode sentences using neural networks,
        # and calculate their cosine similarity in the embedding space (Jin et al., 2020)."
        # The original implementation uses similarity of 0.7.
        use_constraint = UniversalSentenceEncoder(
            threshold=0.7,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        constraints = constraints_in + constraints

        # Goal is untargeted classification.
        # "The score is then the negative probability of predicting the gold label from f, using [x_{adv}] as the input"
        # goal_function = UntargetedClassification(model_wrapper)
        goal_function = goal_fn

        # "To achieve this,  we iteratively apply the actions,
        #  and first select those minimizing the probability of outputting the gold label y from f."
        #
        # "Only one of the three actions can be applied at each position, and we select the one with the highest score."
        #
        # "Actions are iteratively applied to the input, until an adversarial example is found or a limit of actions T
        # is reached.
        #  Each step selects the highest-scoring action from the remaining ones."
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
    
class InputReduction(AttackRecipe):
    """Feng, Wallace, Grissom, Iyyer, Rodriguez, Boyd-Graber. (2018).

    Pathologies of Neural Models Make Interpretations Difficult.

    https://arxiv.org/abs/1804.07781
    """
    @staticmethod
    def build(constraints_in, goal_fn, args):
        # At each step, we remove the word with the lowest importance value until
        # the model changes its prediction.
        transformation = WordDeletion()

        constraints = constraints_in + [RepeatModification(), StopwordModification()]
        #
        # Goal is untargeted classification
        #
        # goal_function = InputReduction(model_wrapper, maximizable=True)
        goal_function = goal_fn
        #
        # "For each word in an input sentence, we measure its importance by the
        # change in the confidence of the original prediction when we remove
        # that word from the sentence."
        #
        # "Instead of looking at the words with high importance values—what
        # interpretation methods commonly do—we take a complementary approach
        # and study how the model behaves when the supposedly unimportant words are
        # removed."
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)