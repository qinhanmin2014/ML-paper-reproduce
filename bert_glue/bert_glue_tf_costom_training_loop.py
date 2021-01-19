import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import argparse
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers.optimization_tf import WarmUp, AdamWeightDecay

parser = argparse.ArgumentParser()
parser.add_argument('-max_seq_length', default=128, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-num_epochs', default=3, type=int)
parser.add_argument('-learning_rate', default=2e-5, type=float)
parser.add_argument('-max_grad_norm', default=1.0, type=float)
parser.add_argument('-warm_up_proportion', default=0.1, type=float)
parser.add_argument('-bert_path', default='bert-base-uncased', type=str)
parser.add_argument('-dataset', default='MRPC', type=str)
parser.add_argument('-report_step', default=100, type=int)
args = parser.parse_args()

if args.dataset in ["MRPC", "RTE", "WNLI", "STS-B"]:
    args.report_step = 10

def load_data(path):
    input_file = open(path, encoding='utf-8')
    if args.dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "STS-B", "MNLI"]:
        lines = input_file.readlines()[1:]
    elif args.dataset in ["CoLA"]:
        lines = input_file.readlines()
    else:
        assert False
    input_file.close()
    input_ids, attention_mask, token_type_ids = [], [], []
    labels = []
    for line in tqdm(lines):
        line_split = line.strip().split("\t")
        if args.dataset == "MRPC":
            assert len(line_split) == 5
            ans = tokenizer.encode_plus(line_split[3], line_split[4], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "QQP":
            assert len(line_split) == 6
            ans = tokenizer.encode_plus(line_split[3], line_split[4], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "SST-2":
            assert len(line_split) == 2
            ans = tokenizer.encode_plus(line_split[0], max_length=args.max_seq_length,
                                        padding="max_length", truncation=True)
        elif args.dataset == "QNLI":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "RTE":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "WNLI":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[1], line_split[2], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "CoLA":
            assert len(line_split) == 4
            ans = tokenizer.encode_plus(line_split[3], max_length=args.max_seq_length,
                                        padding="max_length", truncation=True)
        elif args.dataset == "STS-B":
            assert len(line_split) == 10
            ans = tokenizer.encode_plus(line_split[7], line_split[8], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        elif args.dataset == "MNLI":
            ans = tokenizer.encode_plus(line_split[8], line_split[9], max_length=args.max_seq_length,
                                        padding="max_length", truncation="longest_first")
        else:
            assert False
        input_ids.append(ans.input_ids)
        attention_mask.append(ans.attention_mask)
        token_type_ids.append(ans.token_type_ids)
        if args.dataset == "MRPC":
            labels.append(int(line_split[0]))
        elif args.dataset == "QQP":
            labels.append(int(line_split[5]))
        elif args.dataset == "SST-2":
            labels.append(int(line_split[1]))
        elif args.dataset == "QNLI":
            if line_split[3] == "not_entailment":
                labels.append(0)
            elif line_split[3] == "entailment":
                labels.append(1)
            else:
                assert False
        elif args.dataset == "RTE":
            if line_split[3] == "not_entailment":
                labels.append(0)
            elif line_split[3] == "entailment":
                labels.append(1)
            else:
                assert False
        elif args.dataset == "WNLI":
            labels.append(int(line_split[3]))
        elif args.dataset == "CoLA":
            labels.append(int(line_split[1]))
        elif args.dataset == "STS-B":
            labels.append(float(line_split[9]))
        elif args.dataset == "MNLI":
            if line_split[-1] == "contradiction":
                labels.append(0)
            elif line_split[-1] == "entailment":
                labels.append(1)
            elif line_split[-1] == "neutral":
                labels.append(2)
            else:
                assert False
        else:
            assert False
    return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(labels)

tokenizer = BertTokenizer.from_pretrained(args.bert_path)
if args.dataset == "MRPC":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/MRPC/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/MRPC/dev.tsv")
elif args.dataset == "QQP":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/QQP/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/QQP/dev.tsv")
elif args.dataset == "SST-2":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/SST-2/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/SST-2/dev.tsv")
elif args.dataset == "QNLI":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/QNLI/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/QNLI/dev.tsv")
elif args.dataset == "RTE":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/RTE/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/RTE/dev.tsv")
elif args.dataset == "WNLI":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/WNLI/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/WNLI/dev.tsv")
elif args.dataset == "CoLA":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/CoLA/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/CoLA/dev.tsv")
elif args.dataset == "STS-B":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/STS-B/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/STS-B/dev.tsv")
elif args.dataset == "MNLI":
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data("glue_data/MNLI/train.tsv")
    dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev = load_data("glue_data/MNLI/dev_matched.tsv")
    extra_dev_input_ids, extra_dev_attention_mask, extra_dev_token_type_ids, extra_y_dev = load_data("glue_data/MNLI/dev_mismatched.tsv")
else:
    assert False
    
if args.dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI", "CoLA"]:
    model = TFBertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
elif args.dataset in ["STS-B"]:
    model = TFBertForSequenceClassification.from_pretrained(args.bert_path, num_labels=1)
elif args.dataset in ["MNLI"]:
    model = TFBertForSequenceClassification.from_pretrained(args.bert_path, num_labels=3)
else:
    assert False

num_train_steps = train_input_ids.shape[0] * args.num_epochs // args.batch_size
num_warmup_steps = int(num_train_steps * args.warm_up_proportion)
if args.dataset in ["STS-B"]:
    loss = tf.keras.losses.MeanSquaredError()
else:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=num_train_steps - num_warmup_steps,
    end_learning_rate=0
)
lr_schedule = WarmUp(
    initial_learning_rate=args.learning_rate,
    decay_schedule_fn=lr_schedule,
    warmup_steps=num_warmup_steps
)
optimizer = AdamWeightDecay(
    learning_rate=lr_schedule,
    weight_decay_rate=0.01,
    epsilon=1e-6,
    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    clipnorm=1
    )

train_dataset = tf.data.Dataset.from_tensor_slices((train_input_ids, train_attention_mask, train_token_type_ids, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((dev_input_ids, dev_attention_mask, dev_token_type_ids, y_dev))
val_dataset = val_dataset.batch(args.batch_size)
if args.dataset in ["MNLI"]:
    extra_val_dataset = tf.data.Dataset.from_tensor_slices((extra_dev_input_ids, extra_dev_attention_mask,
                                                            extra_dev_token_type_ids, extra_y_dev))
    extra_val_dataset = extra_val_dataset.batch(args.batch_size)

@tf.function
def train_step(cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y):
    with tf.GradientTape() as tape:
        logits = model([cur_input_ids, cur_attention_mask, cur_token_type_ids], training=True)
        if args.dataset in ["STS-B"]:
            loss_value = loss(cur_y, tf.reshape(logits[0], [-1]))
        else:
            loss_value = loss(cur_y, logits[0])
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

@tf.function
def test_step(cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y):
    logits = model([cur_input_ids, cur_attention_mask, cur_token_type_ids], training=False)
    return logits

total_step = len(train_dataset)
start_time = time.time()
for epoch in range(args.num_epochs):
    print("epoch {}".format(epoch))
    for i, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y) in enumerate(train_dataset):
        loss_value = train_step(cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y)
        if (i + 1) % args.report_step == 0:
            print ('[{}] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, args.num_epochs, i + 1, total_step, float(loss_value)))
    preds = []
    for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in tqdm(val_dataset):
        logits = test_step(cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y)
        if args.dataset in ["STS-B"]:
            preds.extend(list(logits[0].numpy().ravel()))
        else:
            preds.extend(list(np.argmax(logits[0].numpy(), axis=1)))
    if args.dataset in ["MRPC", "QQP", "SST-2", "QNLI", "RTE", "WNLI"]:
        cur_accuracy = accuracy_score(np.array(y_dev), np.array(preds))
        print("accuracy: {:.4f}".format(cur_accuracy))
    if args.dataset in ["MRPC", "QQP"]:
        cur_f1 = f1_score(np.array(y_dev), np.array(preds))
        print("f1: {:.4f}".format(cur_f1))
    if args.dataset in ["CoLA"]:
        cur_matthews = matthews_corrcoef(np.array(y_dev), np.array(preds))
        print("matthews corrcoef: {:.4f}".format(cur_matthews))
    if args.dataset in ["STS-B"]:
        preds = np.clip(np.array(preds), 0, 5)
        cur_pearsonr = pearsonr(np.array(y_dev), preds)[0]
        cur_spearmanr = spearmanr(np.array(y_dev), preds)[0]
        print("pearson corrcoef: {:.4f}".format(cur_pearsonr))
        print("spearman corrcoef: {:.4f}".format(cur_spearmanr))
    if args.dataset in ["MNLI"]:
        cur_accuracy = accuracy_score(np.array(y_dev), np.array(preds))
        print("matched accuracy: {:.4f}".format(cur_accuracy))
        preds = []
        for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in tqdm(extra_val_dataset):
            logits = test_step(cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y)
            preds.extend(list(np.argmax(logits[0].numpy(), axis=1)))
        cur_accuracy = accuracy_score(np.array(extra_y_dev), np.array(preds))
        print("mismatched accuracy: {:.4f}".format(cur_accuracy))
print("training time: {:.4f}".format(time.time() - start_time))