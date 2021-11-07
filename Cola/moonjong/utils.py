import collections
import logging
import os
import pandas as pd
import torch
import numpy as np
import random

import transformers

import yaml


def dict_to_namedtuple(d):
    """
    Convert dictionary to named tuple.
    """
    FLAGSTuple = collections.namedtuple('FLAGS', sorted(d.keys()))

    for k, v in d.items():

        if k == 'prefix':
            v = os.path.join('./', v)

        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)

        elif type(v) is str:
            try:
                d[k] = eval(v)
            except:
                d[k] = v

    nt = FLAGSTuple(**d)

    return nt


def load_yaml(path):
    with open(path, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict


def roberta_base_AdamW_grouped_LLRD(model, init_lr):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]

    for i, (name, params) in enumerate(named_parameters):

        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

        if name.startswith("roberta_model.embeddings") or name.startswith("roberta_model.encoder"):
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr

            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr

            # For set_3, increase lr to 0.0000035
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr

            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})

            # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).
        if name.startswith("regressor") or name.startswith("roberta_model.pooler"):
            lr = init_lr * 3.6

            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})

    return transformers.AdamW(opt_parameters, lr=init_lr)


def set_seed(seed):
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def tokenized_dataset_len(dataset, tokenizer):
    li = []
    for sentence in dataset['sentence']:
        li.append(tokenizer.tokenize(sentence))
    return li


def load_data(dataset_dir):
    dataset = pd.read_csv(dataset_dir, delimiter='\t')
    dataset['acceptability_label'] = dataset['acceptability_label'].astype(int)
    return dataset


def load_test_data(dataset_dir):
    dataset = pd.read_csv(dataset_dir, delimiter='\t')
    return dataset


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def official_f1():
    with open(os.path.join('/content/drive/MyDrive/NIKL/eval/result.txt'), "r", encoding="utf-8") as f:
        macro_result = list(f)[-1]
        macro_result = macro_result.split(":")[1].replace(">>>", "").strip()
        macro_result = macro_result.split("=")[1].strip().replace("%", "")
        macro_result = float(macro_result) / 100

    return macro_result


def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
        # "f1": official_f1(),
    }



def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
