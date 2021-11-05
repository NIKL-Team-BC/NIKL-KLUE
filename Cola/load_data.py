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


def tokenized_dataset(dataset, tokenizer, arch="encoder"):
  sentence = dataset['sentence'].tolist()

  tokenized_sentences = tokenizer(
      sentence,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=150,
      add_special_tokens=True,
      return_token_type_ids = True
      )
      
  return tokenized_sentences
