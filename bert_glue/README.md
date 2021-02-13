# ML-paper-reproduce

reference:
- https://arxiv.org/abs/1810.04805
- https://github.com/google-research/bert
- https://github.com/huggingface/transformers/tree/master/examples/text-classification

CoLA:

```
python bert_glue.py -dataset CoLA
python bert_glue_tf.py -dataset CoLA
python bert_glue_tf_costom_training_loop.py -dataset CoLA
CUDA_VISIBLE_DEVICES=0,1 python bert_glue.py -bert_path bert-large-uncased -dataset CoLA
```

- The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not
- bert paper: Accuracy 52.1%
- transformers: Matthew's corr 56.53%, training time 3:17
- pytorch reproduced (average over 3 seeds): Matthew's corr 57.32%(1.49%), traning time 5:24
- tensorflow reproduced (average over 3 runs): Matthew's corr 58.57%(0.93%), traning time 7:21
- tensorflow custom training loop reproduced (average over 3 runs): Matthew's corr 54.80%(0.88%), traning time 6:39
- bert-large-uncased
  - bert paper: Accuracy 60.5%
  - pytorch reproduced (average over 3 seeds): Matthew's corr 60.44%(2.78%), traning time 13:07

SST-2:

```
python bert_glue.py -dataset SST-2
python bert_glue_tf.py -dataset SST-2
python bert_glue_tf_costom_training_loop.py -dataset SST-2
```

- The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment
- bert paper: Accuracy 93.5%
- transformers: Accuracy 92.32%, training time 26:06
- pytorch reproduced (average over 3 seeds): Accuracy 92.66%(0.34%), traning time 41:24
- tensorflow reproduced (average over 3 runs): Accuracy 92.62%(0.42%), traning time 54:50
- tensorflow custom training loop reproduced (average over 3 runs): Accuracy 92.43%(0.09%), traning time 48:13

STS-B:

```
python bert_glue.py -dataset STS-B
python bert_glue_tf.py -dataset STS-B
python bert_glue_tf_costom_training_loop.py -dataset STS-B
```

- The Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines and other sources. They were annotated with a score from 1 to 5 denoting how similar the two sentences are in terms of semantic meaning.
- bert paper: Spearman corr. 85.8%
- transformers: Person/Spearman corr. 88.64/88.48, training time 2:13
- pytorch reproduced (average over 3 seeds): Person/Spearman corr. 89.29%(0.40%)/88.87%(0.47%), traning time 3:45
- tensorflow reproduced (average over 3 runs): Person/Spearman corr. 88.28%(0.20%)/87.96%(0.23%), traning time 4:41
- tensorflow custom training loop reproduced (average over 3 runs): Person/Spearman corr. 88.14%(0.31%)/87.90%(0.33%), traning time 4:43

MRPC:

```
python bert_glue.py -dataset MRPC
python bert_glue_tf.py -dataset MRPC
python bert_glue_tf_costom_training_loop.py -dataset MRPC
CUDA_VISIBLE_DEVICES=0,1 python bert_glue.py -bert_path bert-large-uncased -dataset MRPC -learning_rate 1e-5
```

- Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent
- bert paper: F1 88.9%
- transformers: F1/Accuracy 88.85%/84.07%, training time 2:21
- pytorch reproduced (average over 3 seeds): F1/Accuracy 90.37%(0.47%)/86.11%(0.76%), traning time 2:18
- tensorflow reproduced (average over 3 runs): F1/Accuracy 88.92%(0.37%)/84.31%(0.72%), traning time 3:20
- tensorflow custom training loop reproduced (average over 3 runs): F1/Accuracy 89.43%(0.54%)/85.05%(0.80%), traning time 2:58
- bert-large-uncased
  - bert paper: F1 89.3%
  - pytorch reproduced (average over 3 seeds): F1/Accuracy 90.13%(0.83%)/86.36%(1.03%), traning time 5:36

QQP:

```
python bert_glue.py -dataset QQP
python bert_glue_tf.py -dataset QQP
python bert_glue_tf_costom_training_loop.py -dataset QQP
```

- Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent
- bert paper: F1 71.2%
- transformers: Accuracy/F1 90.71%/87.49%, training time 2:22:26
- pytorch reproduced (average over 3 seeds): Accuracy/F1 90.92%(0.19%)/87.82%(0.27%), traning time 3:51:28
- tensorflow reproduced (average over 3 runs): Accuracy/F1 91.00%(0.13%)/87.81%(0.21%), traning time 4:58:43
- tensorflow custom training loop reproduced (average over 3 runs): Accuracy/F1 90.91%(0.19%)/87.79%(0.28%), traning time 4:50:44

MNLI:

```
python bert_glue.py -dataset MNLI
python bert_glue_tf.py -dataset MNLI
python bert_glue_tf_costom_training_loop.py -dataset MNLI
```

- Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task. Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one
- bert paper: Matched acc./Mismatched acc. 84.6%/83.4%
- transformers: Matched acc./Mismatched acc. 83.91%/84.10%, training time 2:35:23
- pytorch reproduced (average over 3 seeds): Matched acc./Mismatched acc. 84.46%(0.29%)/84.70(0.29%), traning time 4:17:03
- tensorflow reproduced (average over 3 runs): Matched acc./Mismatched acc. 84.43%(0.35%)/84.54(0.24%), traning time 5:15:48
- tensorflow custom training loop reproduced (average over 3 runs): Matched acc./Mismatched acc. 84.14%(0.26%)/84.46(0.23%), traning time 4:45:34

QNLI:

```
python bert_glue.py -dataset QNLI
python bert_glue_tf.py -dataset QNLI
python bert_glue_tf_costom_training_loop.py -dataset QNLI
```

- Question Natural Language Inference is a version of the Stanford Question Answering Dataset which has been converted to a binary classification task. The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.
- bert paper: Accuracy 90.5%
- transformers: Accuracy 90.66%, training time 40:57
- pytorch reproduced (average over 3 seeds): Accuracy 90.94%(0.77%), traning time 1:05:50
- tensorflow reproduced (average over 3 runs): Accuracy 91.36%(0.23%), traning time 1:19:56
- tensorflow custom training loop reproduced (average over 3 runs): Accuracy 91.23%(0.27%), traning time 1:16:35

RTE:

```
python bert_glue.py -dataset RTE
python bert_glue_tf.py -dataset RTE
python bert_glue_tf_costom_training_loop.py -dataset RTE
```

- Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data
- bert paper: Accuracy 66.4%
- transformers: Accuracy 65.70%, training time 57
- pytorch reproduced (average over 3 seeds): Accuracy 68.59%(2.99%), training time 1:32
- tensorflow reproduced (average over 3 runs): Accuracy 63.54%(1.56%), training time 2:16
- tensorflow custom training loop reproduced (average over 3 runs): Accuracy 64.98%(1.29%), traning time 2:08

WNLI:

```
python bert_glue.py -dataset WNLI -learning_rate 1e-5
python bert_glue_tf.py -dataset WNLI -learning_rate 1e-5
python bert_glue_tf_costom_training_loop.py -dataset WNLI -learning_rate 1e-5
```

- Winograd NLI is a small natural language inference dataset
- transformers: Accuracy 56.34%, training time 24
- pytorch reproduced (seed=0): Accuracy 56.34%, training time 22
- tensorflow reproduced (max over 3 runs): Accuracy 50.70%, training time 51
- tensorflow custom training loop reproduced (max over 3 runs): Accuracy 53.52%, training time 46
- this dataset is too small and the result will change significantly if we use a different seed
