# ML-paper-reproduce

reference:
- https://arxiv.org/abs/1810.04805
- https://github.com/google-research/bert
- https://github.com/huggingface/transformers/tree/master/examples/text-classification

MRPC:

```
python bert_glue.py -report_step 10
```

- Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent
- bert paper: F1 88.9%
- transformers: F1/Accuracy	88.85%/84.07%
- reproduced(average over 3 seeds): F1/Accuracy	90.37% (0.47%) / 86.11% (0.76%)
