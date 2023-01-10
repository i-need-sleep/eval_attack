import transformers
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

import textattack
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")


attack = textattack.attack_recipes.clare_li_2020.CLARE2020.build(model_wrapper)
# Attack 20 samples with CSV logging and checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(num_examples=40, log_to_csv="log.csv", checkpoint_interval=5, checkpoint_dir="checkpoints", disable_stdout=True)
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()