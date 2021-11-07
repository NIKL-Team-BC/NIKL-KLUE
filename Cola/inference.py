import argparse

from torch import nn
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, ElectraForSequenceClassification, AutoConfig, AutoModel
import torch
from tqdm import tqdm
from moonjong.moonjong_model import CoLA
from moonjong.data import convert_test_to_features
from moonjong.utils import load_test_data
from importlib import import_module
from seokmin.utils import check_arch
from seokmin.load_data import *
from seokmin.model import Electra
import json


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=64, shuffle=False)
    model.eval()
    results = []
    preds = []

    for i, items in enumerate(dataloader):
        item = {key: val.to(device) for key, val in items.items()}
        with torch.no_grad():
            outputs = model(**item)
        logits = outputs[0]
        m = nn.Softmax(dim=1)
        logits = m(logits)
        logits = logits.detach().cpu().numpy()  # (Batch_size, 5)  5개의 클래스 확률형태
        pred = logits[:, 1]
        result = np.argmax(logits, axis=-1)
        results += result.tolist()
        preds += pred.tolist()

    return np.array(results).flatten(), np.array(preds).flatten()


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def make_vector_softmax(vector):
    dummy_vector = []
    for i, ele in enumerate(vector):
        dummy_vector.append(softmax(vector[i]))

    return np.array(dummy_vector)


def extract_dohoon_logit(model_path, data_path, tokenizer, device):
    test_data = pd.read_csv(data_path, delimiter='\t', header=0)

    class testDataset(Dataset):
        def __init__(self, dataset, tokenizer, max_len,
                     ):
            self.dataset = dataset['sentence']
            self.sentences = [tokenizer(str(i),
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=max_len,
                                        pad_to_max_length=True,
                                        add_special_tokens=True) for i in self.dataset]

            print(f'{len(self.dataset)}개의 데이터로 init')

        def __getitem__(self, idx):
            input_ids = self.sentences[idx]['input_ids'][0]
            attention_mask = self.sentences[idx]['attention_mask'][0]

            return input_ids, attention_mask

        def __len__(self):
            return (len(self.sentences))

    submission_data = testDataset(test_data, tokenizer, max_len=100)
    model = ElectraForSequenceClassification.from_pretrained('tunib/electra-ko-base')
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    sampler = SequentialSampler(submission_data)
    test_input = DataLoader(submission_data, sampler=sampler, batch_size=64)
    test_output = []
    logit = None
    for batch_id, (token_ids, masked_attention) in enumerate(tqdm(test_input)):
        with torch.no_grad():
            token_ids = token_ids.long().to(device)
            masked_attention = masked_attention.long().to(device)
            out = model(token_ids, masked_attention)[0]

            pred = torch.argmax(out, dim=-1)
            test_output.extend(list(map(int, pred.cpu().numpy())))

            if logit is None:
                logit = out.detach().cpu().numpy()
            else:
                logit = np.append(logit, out.detach().cpu().numpy(), axis=0)
    logit = make_vector_softmax(logit)
    return logit


def extract_moonjong_logit(model_path, data_path, tokenizer, device):
    checkpoint = torch.load(model_path)
    config = AutoConfig.from_pretrained(
        'tunib/electra-ko-base',
        num_labels=2
    )
    model = CoLA('tunib/electra-ko-base', config=config, dropout_rate=0.75)
    model.load_state_dict(checkpoint['model'])

    test_data = load_test_data(data_path)

    test_Dataset = convert_test_to_features(test_data, tokenizer, max_len=50)
    sampler = SequentialSampler(test_Dataset)
    test_dataloader = DataLoader(test_Dataset, sampler=sampler, batch_size=64)

    model.to(device)
    nb_eval_steps = 0
    preds = None
    model.eval()
    for batch in tqdm(test_dataloader, desc="Predicting"):
        batch = tuple(batch[t].to("cuda:0") for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            outputs = model(**inputs)
            # print(outputs)
            pred = outputs[0]

        nb_eval_steps += 1

        if preds is None:
            preds = pred.detach().cpu().numpy()
            # out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, pred.detach().cpu(), axis=0)
            # out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    logit = preds
    logit = make_vector_softmax(logit)
    return logit


def extract_seokmin_logit(args, device):
    TOK_NAME = args.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

    dataset = load_test_data("./cola_data_results/data/NIKL_CoLA_test.tsv")
    test_Dataset = convert_test_to_features(dataset, tokenizer, max_len=150)


    # print(len(test_datase t))
    if len(args.model_dirs) > 1:
        for i, model_dir in enumerate(args.model_dirs, 1):
            model = Electra.from_pretrained(model_dir)
            model.to(device)
            model.eval()

            # predict answer
            pred_answer_, preds_ = inference(model, test_Dataset, device)
            if i == 1:
                pred_answer = pred_answer_.reshape(1, -1)
                preds = preds_.reshape(1, -1)
            else:
                pred_answer = np.concatenate([pred_answer, pred_answer_.reshape(1, -1)], axis=0)
                preds = np.concatenate([preds, preds_.reshape(1, -1)], axis=0)

        preds = preds.mean(axis=0)
        pred_answer = pred_answer.sum(axis=0)
        answer = np.zeros_like(preds)
        answer[pred_answer >= 5] = 1
        pred_answer = answer



    # predict answer
    preds = np.array([[1 - pred, pred] for pred in preds])
    a = 0

    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dirs', type=str,
                        default="./")
    parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-base-v3-discriminator")
    parser.add_argument('--model_type', type=str, default="Electra")
    args = parser.parse_args()
    args.model_dirs = [
        './cola_data_results/results/ElectraV3_4e-06_9k1/1/best',
        './cola_data_results/results/ElectraV3_4e-06_9k2/2/best',
        './cola_data_results/results/ElectraV3_4e-06_9k3/3/best',
        './cola_data_results/results/ElectraV3_4e-06_9k4/4/best',
        './cola_data_results/results/ElectraV3_4e-06_9k5/5/best',
        './cola_data_results/results/ElectraV3_4e-06_9k6/6/best',
        './cola_data_results/results/ElectraV3_4e-06_9k7/7/best',
        './cola_data_results/results/ElectraV3_4e-06_9k8/8/best',
        './cola_data_results/results/ElectraV3_4e-06_9k9/9/best',
    ]

    print(args)


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_path = './test_data/NIKL_CoLA_test.tsv'
    tunib_tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-base')

    moonjong_model_path = './cola_data_results/moonjong/moonjong_cola.pt'
    moonjong_logit = extract_moonjong_logit(moonjong_model_path, data_path, tunib_tokenizer, device)

    dohoon_model_path = './cola_data_results/dohoon/dohoon_cola.pth'
    dohoon_logit = extract_dohoon_logit(dohoon_model_path, data_path, tunib_tokenizer, device)

    seokmin_logit = extract_seokmin_logit(args, device)

    final_logit = (moonjong_logit + dohoon_logit + seokmin_logit).argmax(-1)
    cola_submission = [{'idx': i, 'label': int(label)} for i, label in enumerate(final_logit)]

    cola = {}
    cola['cola'] = cola_submission

    with open('submission.json', 'w') as fb:
        json.dump(cola, fb)