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


def preprocessing_dataset(dataset):
  label = []
  sentence = []
  title = []
  for i in range(len(dataset)):
    if isinstance(dataset.iloc[i]['full_contents'], str):
      label.append((dataset.iloc[i]['r_recommends'], dataset.iloc[i]['r_likes'], dataset.iloc[i]['r_toucheds'], dataset.iloc[i]['r_dislikes'], dataset.iloc[i]['r_sads']))
      sentence.append(dataset.iloc[i]['full_contents'])
      title.append(dataset.iloc[i]['title'])

  out_dataset = pd.DataFrame({'title':title, 'sentence':sentence, 'label':label})
  return out_dataset

def load_data(dataset_dir):
  dataset = pd.read_csv(dataset_dir, delimiter='\t', names=['source', 'acceptability_label', 'source_annotation', 'sentence'], header=0)
  dataset["label"] = dataset["acceptability_label"].astype(int)

  return dataset


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
