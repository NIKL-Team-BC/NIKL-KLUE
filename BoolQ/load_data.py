import pickle as pickle
import os
import pandas as pd
import torch
import re

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


def pre_process(st):
    st = re.sub('\(.*\)|\s-\s.*', '', st)
    st = re.sub('\[.*\]|\s-\s.*', '', st)
    st = st.lower()

    st = re.sub('[”“]', '\"', st)
    st = re.sub('[’‘]', '\'', st)
    st = re.sub('[≫〉》＞』」]', '>', st)
    st = re.sub('[《「『〈≪＜]','<',st)
    st = re.sub('[−–—]', '−', st)
    st = re.sub('[･•・‧]','·', st)
    st = st.replace('／', '/')
    st = st.replace('℃', '도')
    st = st.replace('→', '에서')
    st = st.replace('!', '')
    st = st.replace('，', ',')
    st = st.replace('㎢', 'km')
    st = st.replace('∼', '~')
    st = st.replace('㎜', 'mm')
    st = st.replace('×', '곱하기')
    st = st.replace('=', '는')
    st = st.replace('®', '')
    st = st.replace('㎖', 'ml')
    st = st.replace('ℓ', 'l')
    st = st.replace('˚C', '도')
    st = st.replace('˚', '도')
    st = st.replace('°C', '도')
    st = st.replace('°', '도')
    st = st.replace('＋', '+')
    st = st.replace('*', '')
    st = st.replace(';', '.')
    return st


def load_data(dataset_dir):
    dataset = pd.read_csv(dataset_dir, delimiter='\t', names=['ID', 'text', 'question', 'answer'], header=0)
    dataset["label"] = dataset["answer"].astype(int)
    # dataset['text'] = dataset['text'].apply(pre_process)
    return dataset


# def load_data(dataset_dir):
#     begin_of_sentence = '<C>'
#     begin_of_sentence_q = '<Q>'
#     end_of_sentence = '</C>'
#     end_of_sentence_q = '</Q>'
#     dataset = pd.read_csv(dataset_dir, delimiter='\t', names=['ID', 'text', 'question', 'answer'], header=0)
#     dataset["label"] = dataset["answer"].astype(int)
#     dataset['text'] = dataset['text'].apply(lambda x: begin_of_sentence + x.replace('.', '. '+'[CLS]'))
#     dataset['question'] = dataset['question'].apply(lambda x : '[CLS]'+ begin_of_sentence_q + x)
#     # dataset['text'] = dataset['text'].apply(pre_process)
#     return dataset


def tokenized_dataset(dataset, tokenizer, arch="encoder"):
  sentence = dataset['text'].tolist()
  sentence_d = dataset['question'].tolist()

  if arch=="encoder":
    tokenized_sentences = tokenizer(
        sentence,
        sentence_d,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=350,
        add_special_tokens=True,
        return_token_type_ids = True
        )
  elif arch == "encoder-decoder":
    tokenized_sentences = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=350,
        add_special_tokens=True,
        return_token_type_ids = False
        )
    tokenized_sentences_d = tokenizer(
        sentence_d,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=350,
        add_special_tokens=True,
        return_token_type_ids = False
        )
    for key, value in tokenized_sentences_d.items():
      tokenized_sentences[key+"_d"] = value

  return tokenized_sentences
