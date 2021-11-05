from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import numpy as np
import argparse
from utils import check_arch
import torch.nn as nn

# import argparse
from importlib import import_module

import json

# 한글깨짐 방지
import sys
import io

def load_test_data(dataset_dir):
  dataset = pd.read_csv(dataset_dir, delimiter='\t', names=['ID', 'sentence', 'question', '1', '2','answer'], header=0)
  dataset["label"] = [0] * len(dataset)

  new_sentence1_1 = []
  new_sentence1_2 = []
  new_sentence2_1 = []
  new_sentence2_2 = []
  for i in range(len(dataset)):
    s = dataset.iloc[i]['sentence']
    q = dataset.iloc[i]['question']
    s1 = dataset.iloc[i]['1']
    s2 = dataset.iloc[i]['2']
    lb = dataset.iloc[i]['label']
    if q == "결과":
      new_sentence1_1.append("[결과]" + s)
      new_sentence1_2.append(s1)
      new_sentence2_1.append("[결과]" + s)
      new_sentence2_2.append(s2)

    else:
      new_sentence1_1.append("[원인]" + s1)
      new_sentence1_2.append(s)
      new_sentence2_1.append("[원인]" + s2)
      new_sentence2_2.append(s)

  dataset["new_sentence1_1"] = new_sentence1_1
  dataset["new_sentence1_2"] = new_sentence1_2
  dataset["new_sentence2_1"] = new_sentence2_1
  dataset["new_sentence2_2"] = new_sentence2_2

  return dataset

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=8, shuffle=False)
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
    logits = logits.detach().cpu().numpy()   # (Batch_size, 5)  5개의 클래스 확률형태
    pred = logits[:,1]
    result = np.argmax(logits, axis=-1)
    results += result.tolist()
    preds += pred.tolist()

  return np.array(results).flatten(), np.array(preds).flatten()


def main(args):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
  # load my model 
  model_module = getattr(import_module("model"), args.model_type)
  dataset = load_test_data("./copa_data_results/data/SKT_COPA_Test.tsv")
  test_label = dataset['label'].values
  tokenized_test = tokenized_dataset(dataset, tokenizer, check_arch(args.model_type))
  test_dataset = CustomDataset(tokenized_test, test_label)

  print(len(test_dataset))

  if len(args.model_dirs) > 1:
    for i, model_dir in enumerate(args.model_dirs, 1):
      model = model_module.from_pretrained(model_dir, args=args)  # args 기존으로 돌리려면 빼줘야 함
      model.parameters
      model.to(device)
      model.eval()

      # predict answer
      pred_answer_, preds_ = inference(model, test_dataset, device)
      if i == 1 :
        pred_answer = pred_answer_.reshape(1,-1)
        preds = preds_.reshape(1,-1)
      else:
        pred_answer = np.concatenate([pred_answer, pred_answer_.reshape(1,-1)], axis=0)
        preds = np.concatenate([preds, preds_.reshape(1,-1)], axis=0)
    preds = preds.mean(axis=0)
    pred_answer = np.zeros_like(preds)
    pred_answer[preds >= 0.5] = 1
  else:
    model = model_module.from_pretrained(args.model_dirs[0], args=args)  
    model.parameters
    model.to(device)
    model.eval()

    # predict answer
    pred_answer, preds = inference(model, test_dataset, device)

  # make csv file with predicted answer
  dataset = pd.read_csv("./copa_data_results/data/SKT_COPA_Test.tsv", delimiter='\t', names=['ID', 'sentence', 'question', '1', '2','answer'], header=0)
  dataset["model_answer"] = pred_answer
  dataset["model_pred"] = preds
  dataset.to_csv(args.outpath, index=False, encoding="utf-8-sig")

  # make json
  submission_json = {"copa" : []}
  for i, pred in enumerate(pred_answer.tolist()):
    submission_json["copa"].append({"idx" : i, "label" : int(pred+1)})
  with open('submission.json', 'w') as fp:
    json.dump(submission_json, fp)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
 
  # model dir
  parser.add_argument('--model_dirs', type=str, default="/de")
  parser.add_argument('--outpath', type=str, default="./submission.csv")
  parser.add_argument('--model_type', type=str, default="Roberta")
  parser.add_argument('--pretrained_model', type=str, default="klue/roberta-large")
  parser.add_argument('--dropout_rate', type=float, default=0, help="Dropout for fully-connected layers")
  args = parser.parse_args()

  args.model_type = "Roberta"
  args.pretrained_model = "klue/roberta-large"
  args.model_dirs = [
    './copa_data_results/results/Best_AddData_Roberta_8e-06/1/best',
    './copa_data_results/results/BEST_Roberta5_Fz0_drop02_8e-06__test/1/best',
    ]


  print(args)
  main(args)
