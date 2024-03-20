# Code and data for the paper [Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks](https://arxiv.org/abs/2311.00508) (accepted in EMNLP 2023 Findings).

## Human Annotations
The human ratings and perturbed sentence pairs is available through [Google Drive](https://drive.google.com/file/d/1JjWbTGpQBYZwXI29iojoOBGabxR9kb4h/view?usp=sharing).

## Code
* Process WMT data and evaluate the original sentence pairs with [data_preproc.py](https://github.com/i-need-sleep/eval_attack/blob/main/code/data_preproc.py).
* Apply attacks with [attack.py](https://github.com/i-need-sleep/eval_attack/blob/main/code/attack.py).

## Citation
```
  @inproceedings{huang-baldwin-2023-robustness,
      title = "Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks",
      author = "Huang, Yichen  and
        Baldwin, Timothy",
      editor = "Bouamor, Houda  and
        Pino, Juan  and
        Bali, Kalika",
      booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
      month = dec,
      year = "2023",
      address = "Singapore",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2023.findings-emnlp.340",
      doi = "10.18653/v1/2023.findings-emnlp.340",
      pages = "5126--5135",
      abstract = "We investigate MT evaluation metric performance on adversarially-synthesized texts, to shed light on metric robustness. We experiment with word- and character-level attacks on three popular machine translation metrics: BERTScore, BLEURT, and COMET. Our human experiments validate that automatic metrics tend to overpenalize adversarially-degraded translations. We also identify inconsistencies in BERTScore ratings, where it judges the original sentence and the adversarially-degraded one as similar, while judging the degraded translation as notably worse than the original with respect to the reference. We identify patterns of brittleness that motivate more robust metric development.",
  }
```
