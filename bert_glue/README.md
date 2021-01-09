# ML-paper-reproduce

reference:
- https://arxiv.org/abs/1810.04805
- https://github.com/google-research/bert
- https://github.com/huggingface/transformers/tree/master/examples/text-classification

CoLA:

```
python bert_glue.py -dataset CoLA
```

- The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not
- bert paper: Accuracy 52.1%
- transformers: Matthew's corr 56.53%, training time 3:17
- reproduced (average over 3 seeds): Matthew's corr 58.11%(2.27%), traning time 3:25

MRPC:

```
python bert_glue.py -dataset MRPC
```

- Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent
- bert paper: F1 88.9%
- transformers: F1/Accuracy 88.85%/84.07%, training time 2:21
- reproduced (average over 3 seeds): F1/Accuracy 90.37%(0.47%)/86.11%(0.76%), traning time 2:18

RTE:

```
python bert_glue.py -dataset RTE
```

- Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with
much less training data
- bert paper: Accuracy 66.4%
- transformers: Accuracy 65.70%, training time 57s
- reproduced (average over 3 seeds): Accuracy 68.59%(2.99%), training time 1:32

WNLI:

```
python bert_glue.py -dataset WNLI -learning_rate 1e-5
```

- Winograd NLI is a small natural language inference dataset
- transformers: Accuracy 56.34%, training time 24s
- reproduced (seed=0): Accuracy 56.34%, training time 22s
- this dataset is too small and the result will change significantly if we use a different seed