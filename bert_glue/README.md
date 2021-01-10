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

SST-2:

```
python bert_glue.py -dataset SST-2
```

- The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment
- bert paper: Accuracy 93.5%
- transformers: Accuracy 92.32%, training time 26:06
- reproduced (average over 3 seeds): Accuracy 92.66%(0.34%), traning time 41:24

MRPC:

```
python bert_glue.py -dataset MRPC
```

- Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent
- bert paper: F1 88.9%
- transformers: F1/Accuracy 88.85%/84.07%, training time 2:21
- reproduced (average over 3 seeds): F1/Accuracy 90.37%(0.47%)/86.11%(0.76%), traning time 2:18

QQP:

```
python bert_glue.py -dataset QQP
```

- Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent
- bert paper: F1 71.2%
- transformers: Accuracy/F1 90.71%/87.49%, training time 2:22:26
- reproduced (average over 3 seeds): Accuracy/F1 90.92%(0.19%)/87.82%(0.27%), traning time 3:51:28

QNLI:

```
python bert_glue.py -dataset QNLI
```

- Question Natural Language Inference is a version of the Stanford Question Answering Dataset which has been converted to a binary classification task. The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.
- bert paper: Accuracy 90.5%
- transformers: Accuracy 90.66%, training time 40:57
- reproduced (average over 3 seeds): Accuracy 90.94%(0.77%), traning time 1:05:50

RTE:

```
python bert_glue.py -dataset RTE
```

- Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data
- bert paper: Accuracy 66.4%
- transformers: Accuracy 65.70%, training time 57
- reproduced (average over 3 seeds): Accuracy 68.59%(2.99%), training time 1:32

WNLI:

```
python bert_glue.py -dataset WNLI -learning_rate 1e-5
```

- Winograd NLI is a small natural language inference dataset
- transformers: Accuracy 56.34%, training time 24
- reproduced (seed=0): Accuracy 56.34%, training time 22
- this dataset is too small and the result will change significantly if we use a different seed
