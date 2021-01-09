# paper: https://arxiv.org/abs/1810.04805
# data: https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003
# reported dev 96.4
# reported test 92.4
# reproduced dev (average over 10 runs) : 96.44 (0.08)
# reproduced test (average over 10 runs) : 92.04 (0.32)

import torch
print(torch.__version__)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import re
import random
random.seed(1)
import numpy as np
np.random.seed(1)
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertForTokenClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from seqeval.metrics import f1_score
import truecase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


max_seq_length = 192
padding_index = -100
batch_size = 32
learning_rate = 5e-5
num_epochs = 5
max_grad_norm = 1.0
warm_up_proportion = 0.1
dataset = "conll2003"


conll03_labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
conll03_labels_dict = {}
for i in range(len(conll03_labels)):
    conll03_labels_dict[conll03_labels[i]] = i


tokenizer = BertTokenizer.from_pretrained("../bert-base-cased", do_lower_case=False)
cls_index = tokenizer.convert_tokens_to_ids("[CLS]")
sep_index = tokenizer.convert_tokens_to_ids("[SEP]")
pad_index = tokenizer.convert_tokens_to_ids("[PAD]")


def truecase_sentence(tokens):
    previous_len = len(tokens)
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z]+\b', w)]
    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()
        if len(parts) != len(word_lst):
            return tokens
        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    assert len(tokens) == previous_len
    return tokens


def process(input_file, is_train):
    with open(input_file) as f:
        lines = f.readlines()
    input_ids = []
    labels = []
    cur_input_ids = []
    cur_labels = []
    for line in tqdm(lines):
        if line.startswith("-DOCSTART-") or len(line.strip()) == 0:
            assert len(cur_input_ids) == len(cur_labels)
            if len(cur_input_ids) != 0:
                input_ids.append(cur_input_ids)
                labels.append(cur_labels)
                cur_input_ids = []
                cur_labels = []
        else:
            splits = line.strip().split(" ")
            assert len(splits) == 4
            cur_word = splits[0]
            cur_label = splits[3]
            cur_input_ids.append(cur_word)
            cur_labels.append(cur_label)
    assert len(cur_input_ids) == len(cur_labels)
    if len(cur_input_ids) != 0:
        input_ids.append(cur_input_ids)
        labels.append(cur_labels)
    for i in tqdm(range(len(input_ids))):
        input_ids[i] = truecase_sentence(input_ids[i])
        cur_input_ids = []
        cur_labels = []
        assert len(input_ids[i]) == len(labels[i])
        for j in range(len(input_ids[i])):
            cur_word = tokenizer.tokenize(input_ids[i][j])
            cur_word = tokenizer.convert_tokens_to_ids(cur_word)
            if len(cur_word) >= 1:
                cur_input_ids.extend(cur_word)
                cur_labels.extend([conll03_labels_dict[labels[i][j]]] + [padding_index] * (len(cur_word) - 1))
            else:
                assert False  # TODO: more logic here
        input_ids[i] = cur_input_ids
        labels[i] = cur_labels
    attention_mask = []
    token_type_ids = []
    for i in range(len(input_ids)):
        if len(input_ids[i]) > max_seq_length - 2:
            print("sequence too long", len(input_ids[i]))
            if is_train:
                input_ids[i] = input_ids[i][:max_seq_length - 2]
                labels[i] = labels[i][:max_seq_length - 2]
            else:
                assert False  # TODO: more logic here
        input_ids[i] = [cls_index] + input_ids[i] + [sep_index]
        labels[i] = [padding_index] + labels[i] + [padding_index]
        attention_mask.append([1] * len(input_ids[i]) + [0] * (max_seq_length - len(input_ids[i])))
        token_type_ids.append([0] * max_seq_length)
        labels[i] = labels[i] + [padding_index] * (max_seq_length - len(input_ids[i]))
        input_ids[i] = input_ids[i] + [pad_index] * (max_seq_length - len(input_ids[i]))
        if not len(input_ids[i]) == len(attention_mask[i]) == len(token_type_ids[i]) == len(labels[i]) == max_seq_length:
            print(len(input_ids[i]), len(attention_mask[i]), len(token_type_ids[i]), len(labels[i]))
    return input_ids, attention_mask, token_type_ids, labels


if dataset == "conll2003":
    train_input_ids, train_attention_mask, train_token_type_ids, train_labels = process("data/CoNLL2003/train.txt", is_train=True)
    dev_input_ids, dev_attention_mask, dev_token_type_ids, dev_labels = process("data/CoNLL2003/dev.txt", is_train=False)
    test_input_ids, test_attention_mask, test_token_type_ids, test_labels = process("data/CoNLL2003/test.txt", is_train=False)
elif dataset == "conll2003pp":
    train_input_ids, train_attention_mask, train_token_type_ids, train_labels = process("data/CoNLL2003_pp/conllpp_train.txt", is_train=True)
    dev_input_ids, dev_attention_mask, dev_token_type_ids, dev_labels = process("data/CoNLL2003_pp/conllpp_dev.txt", is_train=False)
    test_input_ids, test_attention_mask, test_token_type_ids, test_labels = process("data/CoNLL2003_pp/conllpp_test.txt", is_train=False)
else:
    assert False


print(train_input_ids[0][:20])
print(train_attention_mask[0][:20])
print(train_labels[0][:20])


model = BertForTokenClassification.from_pretrained("../bert-base-cased", num_labels=len(conll03_labels))
model.to(device);


train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.long)
train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.long)
dev_input_ids = torch.tensor(dev_input_ids, dtype=torch.long)
dev_attention_mask = torch.tensor(dev_attention_mask, dtype=torch.long)
dev_token_type_ids = torch.tensor(dev_token_type_ids, dtype=torch.long)
dev_labels = torch.tensor(dev_labels, dtype=torch.long)
test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
test_attention_mask = torch.tensor(test_attention_mask, dtype=torch.long)
test_token_type_ids = torch.tensor(test_token_type_ids, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)


train_data = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dev_data = TensorDataset(dev_input_ids, dev_attention_mask, dev_token_type_ids, dev_labels)
dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_token_type_ids, test_labels)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


dev_scores = []
test_scores = []
# param_optimizer = list(model.named_parameters())
# no_decay = ['bias','LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
# optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(len(train_loader) * num_epochs * warm_up_proportion),
                num_training_steps=len(train_loader) * num_epochs)
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_labels) in enumerate(train_loader):
        cur_input_ids = cur_input_ids.to(device)
        cur_attention_mask = cur_attention_mask.to(device)
        cur_token_type_ids = cur_token_type_ids.to(device)
        cur_labels = cur_labels.to(device)
        outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(outputs[0].view(-1, len(conll03_labels)), cur_labels.view(-1))
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        if (i + 1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    model.eval()
    dev_y_true = []
    dev_y_pred = []
    with torch.no_grad():
        for i, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_labels) in enumerate(dev_loader):
            cur_input_ids = cur_input_ids.to(device)
            cur_attention_mask = cur_attention_mask.to(device)
            cur_token_type_ids = cur_token_type_ids.to(device)
            cur_labels = cur_labels.to(device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            _, predicted = torch.max(outputs[0], 2)
            dev_y_true.extend(list(np.array(cur_labels.view(-1).cpu())))
            dev_y_pred.extend(list(np.array(predicted.view(-1).cpu())))
    dev_y_pred = list(np.array(conll03_labels)[np.array(dev_y_pred)[np.array(dev_y_true) != -100]])
    dev_y_true = list(np.array(conll03_labels)[np.array(dev_y_true)[np.array(dev_y_true) != -100]])
    print("dev_score: ", f1_score(dev_y_true, dev_y_pred))
    dev_scores.append(f1_score(dev_y_true, dev_y_pred))
    test_y_true = []
    test_y_pred = []
    with torch.no_grad():
        for i, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_labels) in enumerate(test_loader):
            cur_input_ids = cur_input_ids.to(device)
            cur_attention_mask = cur_attention_mask.to(device)
            cur_token_type_ids = cur_token_type_ids.to(device)
            cur_labels = cur_labels.to(device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            _, predicted = torch.max(outputs[0], 2)
            test_y_true.extend(list(np.array(cur_labels.view(-1).cpu())))
            test_y_pred.extend(list(np.array(predicted.view(-1).cpu())))
    test_y_pred = list(np.array(conll03_labels)[np.array(test_y_pred)[np.array(test_y_true) != -100]])
    test_y_true = list(np.array(conll03_labels)[np.array(test_y_true)[np.array(test_y_true) != -100]])
    print("test_score: ", f1_score(test_y_true, test_y_pred))
    test_scores.append(f1_score(test_y_true, test_y_pred))


print(np.argmax(dev_scores), dev_scores[np.argmax(dev_scores)], test_scores[np.argmax(dev_scores)])
