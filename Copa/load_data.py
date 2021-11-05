import pickle as pickle
import os
import pandas as pd
import torch

# Dataset 구성.
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def load_data(dataset_dir):
  dataset = pd.read_csv(dataset_dir, delimiter='\t', names=['ID', 'sentence', 'question', '1', '2','answer'], header=0)
  dataset["label"] = dataset["answer"].astype(int) - 1

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
      # new_sentence1_1.append(s)
      new_sentence1_2.append(s1)
      new_sentence2_1.append("[결과]" + s)
      # new_sentence2_1.append(s)
      new_sentence2_2.append(s2)

    else:
      new_sentence1_1.append("[원인]" + s1)
      # new_sentence1_1.append(s1)
      new_sentence1_2.append(s)
      new_sentence2_1.append("[원인]" + s2)
      # new_sentence2_1.append(s2)
      new_sentence2_2.append(s)

  dataset["new_sentence1_1"] = new_sentence1_1
  dataset["new_sentence1_2"] = new_sentence1_2
  dataset["new_sentence2_1"] = new_sentence2_1
  dataset["new_sentence2_2"] = new_sentence2_2

  return dataset


def tokenized_dataset(dataset, tokenizer, arch="encoder"):
  sentence1_1 = dataset['new_sentence1_1'].tolist()
  sentence1_2 = dataset['new_sentence1_2'].tolist()
  sentence2_1 = dataset["new_sentence2_1"].tolist()
  sentence2_2 = dataset["new_sentence2_2"].tolist()

  tokenized_sentences = tokenizer(
      sentence1_1,
      sentence1_2,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=150,
      add_special_tokens=True,
      return_token_type_ids = True
      )
  tokenized_sentences2 = tokenizer(
      sentence2_1,
      sentence2_2,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=150,
      add_special_tokens=True,
      return_token_type_ids = True
      )
  for key, value in tokenized_sentences2.items():
    tokenized_sentences[key+"2"] = value

  return tokenized_sentences
